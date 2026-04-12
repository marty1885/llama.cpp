// rwkv_probe/root_io.h — ROOT I/O for the prompt/states tree format.
//
// Schema is locked to the existing format used by all prior runs:
//   <prompt_id>.root
//     prompt/                              (TDirectory)
//       prompt_text   (TNamed)
//       prompt_id     (TNamed)
//       n_layer       (TNamed)
//       n_embd_s      (TNamed)
//       states        (TTree)
//         phase/I
//         r_att_L<il>[n_half]/F            for il in [0, n_layer)
//         r_ffn_L<il>[n_half]/F            for il in [0, n_layer)
//         s_L<il>[n_embd_s]/F              for il in [0, n_layer)
//
// Reader: lifts load_patch_source from the original rwkv-probe.cpp (104-183).
// Writer: lifts the inline TFile/TTree branch setup (658-688, 752-760), with
// SetAutoFlush set so memory stays bounded for long runs (the user's RAM
// constraint).
#pragma once

#include "model.h"
#include "state.h"

#include <string>
#include <vector>

class TFile;
class TTree;
class TDirectory;

namespace rwkv_probe { namespace root_io {

// Read a single phase entry from a previous run's ROOT file. Returns true on
// success and fills `out`. Returns false (with a stderr warning) if the file,
// directory, tree, or requested phase is missing.
//
// `out` must already be sized for the target geometry (i.e. constructed with
// the same ModelGeometry as `geom`).
bool read_state(const std::string & path, int phase,
                const ModelGeometry & geom, StateBuf & out);

// Streaming writer: one TFile per prompt, holds the TTree open and writes
// rows incrementally. Bounded memory via SetAutoFlush.
class StateWriter {
public:
    StateWriter(const std::string & path,
                const std::string & prompt_id,
                const std::string & prompt_text,
                const ModelGeometry & geom);
    ~StateWriter();

    StateWriter(const StateWriter &)             = delete;
    StateWriter & operator=(const StateWriter &) = delete;

    // copies state into the per-layer ROOT staging buffers and Fill()s the tree.
    // Throws on geometry mismatch or I/O error.
    void write_row(int phase, const StateBuf & state);

    // closes the file. Called automatically by the destructor; also exposed
    // so experiments can flush before continuing if they want.
    void close();

private:
    ModelGeometry m_geom;
    bool          m_open = false;

    TFile *       m_file = nullptr;
    TDirectory *  m_dir  = nullptr;
    TTree *       m_tree = nullptr;

    int           m_phase_buf = 0;

    // per-layer staging buffers; ROOT branches address into these.
    std::vector<std::vector<float>> m_r_att;
    std::vector<std::vector<float>> m_r_ffn;
    std::vector<std::vector<float>> m_s_wkv;
};

}}  // namespace rwkv_probe::root_io
