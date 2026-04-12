// rwkv_probe/root_io.cpp — ROOT I/O implementation.
#include "rwkv_probe/root_io.h"

#include "TFile.h"
#include "TNamed.h"
#include "TTree.h"
#include "TDirectory.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace rwkv_probe { namespace root_io {

// ── reader (lifts load_patch_source) ──────────────────────────────────────────
bool read_state(const std::string & path, int wanted_phase,
                const ModelGeometry & geom, StateBuf & out) {
    if (out.geom() != geom) {
        die("root_io::read_state: out StateBuf geometry does not match `geom` argument");
    }

    TFile * f = TFile::Open(path.c_str(), "READ");
    if (!f || f->IsZombie()) {
        std::fprintf(stderr, "root_io::read_state: cannot open %s\n", path.c_str());
        if (f) { f->Close(); delete f; }
        return false;
    }

    TDirectory * dir = f->GetDirectory("prompt");
    if (!dir) {
        std::fprintf(stderr, "root_io::read_state: missing prompt/ in %s\n", path.c_str());
        f->Close(); delete f;
        return false;
    }
    TTree * tree = dynamic_cast<TTree *>(dir->Get("states"));
    if (!tree) {
        std::fprintf(stderr, "root_io::read_state: missing prompt/states tree in %s\n",
                     path.c_str());
        f->Close(); delete f;
        return false;
    }

    const int n_layer  = geom.n_layer;
    const int n_half   = geom.n_half;
    const int n_embd_s = geom.n_embd_s;

    // per-layer staging (TTree::SetBranchAddress wants stable pointers)
    std::vector<std::vector<float>> r_att(n_layer, std::vector<float>(n_half));
    std::vector<std::vector<float>> r_ffn(n_layer, std::vector<float>(n_half));
    std::vector<std::vector<float>> s_wkv(n_layer, std::vector<float>(n_embd_s));

    Int_t phase_buf = -1;
    tree->SetBranchAddress("phase", &phase_buf);
    bool addr_ok = true;
    for (int il = 0; il < n_layer; ++il) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "r_att_L%d", il);
        if (tree->SetBranchAddress(nm, r_att[il].data()) < 0) {
            std::fprintf(stderr, "root_io::read_state: missing branch %s\n", nm);
            addr_ok = false; break;
        }
        std::snprintf(nm, sizeof nm, "r_ffn_L%d", il);
        if (tree->SetBranchAddress(nm, r_ffn[il].data()) < 0) {
            std::fprintf(stderr, "root_io::read_state: missing branch %s\n", nm);
            addr_ok = false; break;
        }
        std::snprintf(nm, sizeof nm, "s_L%d", il);
        if (tree->SetBranchAddress(nm, s_wkv[il].data()) < 0) {
            std::fprintf(stderr, "root_io::read_state: missing branch %s\n", nm);
            addr_ok = false; break;
        }
    }
    if (!addr_ok) {
        f->Close(); delete f;
        return false;
    }

    bool found = false;
    Long64_t n_entries = tree->GetEntries();
    for (Long64_t i = 0; i < n_entries; ++i) {
        if (tree->GetEntry(i) <= 0) continue;
        if (phase_buf == wanted_phase) { found = true; break; }
    }
    if (!found) {
        std::fprintf(stderr, "root_io::read_state: phase=%d not found in %s (have %lld entries)\n",
                     wanted_phase, path.c_str(), (long long) n_entries);
        f->Close(); delete f;
        return false;
    }

    // pack the per-layer staging back into the StateBuf flat layout
    for (int il = 0; il < n_layer; ++il) {
        // r_att = first half, r_ffn = second half
        auto rat = out.r_att(il);
        auto rfn = out.r_ffn(il);
        auto sw  = out.s_wkv(il);
        for (int j = 0; j < n_half;   ++j) rat[j] = r_att[il][j];
        for (int j = 0; j < n_half;   ++j) rfn[j] = r_ffn[il][j];
        for (int j = 0; j < n_embd_s; ++j) sw[j]  = s_wkv[il][j];
    }

    f->Close();
    delete f;
    std::fprintf(stderr, "root_io::read_state: loaded state from %s (phase=%d)\n",
                 path.c_str(), wanted_phase);
    return true;
}

// ── writer (lifts inline TFile/TTree setup) ───────────────────────────────────
StateWriter::StateWriter(const std::string & path,
                         const std::string & prompt_id,
                         const std::string & prompt_text,
                         const ModelGeometry & geom)
    : m_geom(geom) {
    m_file = TFile::Open(path.c_str(), "RECREATE");
    if (!m_file || m_file->IsZombie()) {
        die("root_io::StateWriter: cannot open " + path);
    }

    m_dir = m_file->mkdir("prompt");
    if (!m_dir) {
        m_file->Close(); delete m_file; m_file = nullptr;
        die("root_io::StateWriter: cannot mkdir prompt/ in " + path);
    }
    m_dir->cd();

    TNamed("prompt_text", prompt_text.c_str()).Write();
    TNamed("prompt_id",   prompt_id.c_str()).Write();
    TNamed("n_layer",     std::to_string(geom.n_layer).c_str()).Write();
    TNamed("n_embd_s",    std::to_string(geom.n_embd_s).c_str()).Write();

    m_tree = new TTree("states", "RWKV state: prompt-end + EOG");
    m_tree->SetAutoSave(0);
    // bound the basket memory: flush to disk every ~16 MB worth of rows.
    m_tree->SetAutoFlush(-16 * 1024 * 1024);

    m_r_att.assign(geom.n_layer, std::vector<float>(geom.n_half, 0.0f));
    m_r_ffn.assign(geom.n_layer, std::vector<float>(geom.n_half, 0.0f));
    m_s_wkv.assign(geom.n_layer, std::vector<float>(geom.n_embd_s, 0.0f));

    m_tree->Branch("phase", &m_phase_buf, "phase/I");
    char nm[64], spec[64];
    for (int il = 0; il < geom.n_layer; ++il) {
        std::snprintf(nm,   sizeof nm,   "r_att_L%d", il);
        std::snprintf(spec, sizeof spec, "r_att_L%d[%d]/F", il, geom.n_half);
        m_tree->Branch(nm, m_r_att[il].data(), spec);

        std::snprintf(nm,   sizeof nm,   "r_ffn_L%d", il);
        std::snprintf(spec, sizeof spec, "r_ffn_L%d[%d]/F", il, geom.n_half);
        m_tree->Branch(nm, m_r_ffn[il].data(), spec);

        std::snprintf(nm,   sizeof nm,   "s_L%d", il);
        std::snprintf(spec, sizeof spec, "s_L%d[%d]/F", il, geom.n_embd_s);
        m_tree->Branch(nm, m_s_wkv[il].data(), spec);
    }

    m_open = true;
}

StateWriter::~StateWriter() {
    if (m_open) {
        try { close(); } catch (...) { /* swallow in dtor */ }
    }
}

void StateWriter::write_row(int phase, const StateBuf & state) {
    if (!m_open) {
        die("root_io::StateWriter::write_row called after close()");
    }
    if (state.geom() != m_geom) {
        die("root_io::StateWriter::write_row: state geometry does not match writer geometry");
    }

    const int n_half   = m_geom.n_half;
    const int n_embd_s = m_geom.n_embd_s;

    for (int il = 0; il < m_geom.n_layer; ++il) {
        auto rat = state.r_att(il);
        auto rfn = state.r_ffn(il);
        auto sw  = state.s_wkv(il);
        for (int j = 0; j < n_half;   ++j) m_r_att[il][j] = rat[j];
        for (int j = 0; j < n_half;   ++j) m_r_ffn[il][j] = rfn[j];
        for (int j = 0; j < n_embd_s; ++j) m_s_wkv[il][j] = sw[j];
    }

    m_phase_buf = phase;
    m_tree->Fill();
}

void StateWriter::close() {
    if (!m_open) return;

    if (m_dir)  m_dir->cd();
    if (m_tree) m_tree->Write("", /*TObject::kOverwrite=*/2);
    if (m_file) {
        m_file->Close();
        delete m_file;
        m_file = nullptr;
    }
    m_dir  = nullptr;
    m_tree = nullptr;
    m_open = false;
}

}}  // namespace rwkv_probe::root_io
