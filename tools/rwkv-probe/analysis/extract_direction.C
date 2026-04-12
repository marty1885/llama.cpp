// extract_direction.C — extract mean-diff direction from decomposed residual data.
//
// Computes direction = normalize(mean_dangerous - mean_safe) for a given
// stream (e.g. "l_out") at a given layer, and saves it as a binary float32 file.
//
// Usage:
//   root -l -b -q 'extract_direction.C+("build/decompose_500.root", "l_out", 13, "refusal_dir_L13.bin")'

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

void extract_direction(
    const char * file    = "decompose_500.root",
    const char * stream  = "l_out",
    int          layer   = 13,
    const char * outfile = "refusal_dir.bin")
{
    auto * fin = TFile::Open(file);
    if (!fin || fin->IsZombie()) { std::cerr << "cannot open " << file << "\n"; return; }
    auto * tree = dynamic_cast<TTree *>(fin->Get("decompose"));
    if (!tree) { std::cerr << "no decompose tree\n"; return; }

    int n = tree->GetEntries();

    // detect n_embd
    TString br_name = TString::Format("%s_L%d", stream, layer);
    auto * br = tree->GetBranch(br_name.Data());
    if (!br) { std::cerr << "no branch " << br_name << "\n"; return; }
    int n_embd = br->GetLeaf(br_name.Data())->GetLen();

    // load labels
    std::vector<int> labels(n);
    {
        Int_t buf;
        tree->SetBranchAddress("label", &buf);
        for (int i = 0; i < n; ++i) { tree->GetEntry(i); labels[i] = buf; }
        tree->ResetBranchAddresses();
    }

    int n_safe = std::count(labels.begin(), labels.end(), 0);
    int n_dang = std::count(labels.begin(), labels.end(), 1);

    // compute class means
    std::vector<double> mean_safe(n_embd, 0.0);
    std::vector<double> mean_dang(n_embd, 0.0);
    std::vector<float> row(n_embd);

    tree->SetBranchAddress(br_name.Data(), row.data());
    for (int i = 0; i < n; ++i) {
        tree->GetEntry(i);
        if (labels[i] == 0) {
            for (int j = 0; j < n_embd; ++j) mean_safe[j] += row[j];
        } else {
            for (int j = 0; j < n_embd; ++j) mean_dang[j] += row[j];
        }
    }
    tree->ResetBranchAddresses();

    for (int j = 0; j < n_embd; ++j) mean_safe[j] /= n_safe;
    for (int j = 0; j < n_embd; ++j) mean_dang[j] /= n_dang;

    // direction = mean_dang - mean_safe, normalized
    std::vector<float> dir(n_embd);
    double norm = 0;
    for (int j = 0; j < n_embd; ++j) {
        dir[j] = (float)(mean_dang[j] - mean_safe[j]);
        norm += (double)dir[j] * dir[j];
    }
    norm = std::sqrt(norm);
    for (int j = 0; j < n_embd; ++j) dir[j] /= (float)norm;

    // save as binary float32
    std::ofstream ofs(outfile, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(dir.data()), n_embd * sizeof(float));
    ofs.close();

    // also compute separation metric
    double proj_safe = 0, proj_dang = 0;
    for (int j = 0; j < n_embd; ++j) {
        proj_safe += mean_safe[j] * dir[j];
        proj_dang += mean_dang[j] * dir[j];
    }

    std::cout << "stream:    " << stream << "\n";
    std::cout << "layer:     " << layer << "\n";
    std::cout << "n_embd:    " << n_embd << "\n";
    std::cout << "n_safe:    " << n_safe << "\n";
    std::cout << "n_dang:    " << n_dang << "\n";
    std::cout << "||dir||:   " << norm << " (before normalization)\n";
    std::cout << "proj_safe: " << proj_safe << "\n";
    std::cout << "proj_dang: " << proj_dang << "\n";
    std::cout << "gap:       " << (proj_dang - proj_safe) << "\n";
    std::cout << "saved:     " << outfile << " (" << n_embd << " x float32 = "
              << n_embd * 4 << " bytes)\n";

    fin->Close();
}
