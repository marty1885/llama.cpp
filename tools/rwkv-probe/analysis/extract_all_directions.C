// extract_all_directions.C — extract mean-diff directions at all layers.
//
// Usage:
//   root -l -b -q 'extract_all_directions.C+("build/decompose_500.root", "l_out", 0, 31, "build/directions")'

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

void extract_all_directions(
    const char * file    = "decompose_500.root",
    const char * stream  = "l_out",
    int          layer_lo = 0,
    int          layer_hi = 31,
    const char * out_dir = "directions")
{
    auto * fin = TFile::Open(file);
    if (!fin || fin->IsZombie()) { std::cerr << "cannot open " << file << "\n"; return; }
    auto * tree = dynamic_cast<TTree *>(fin->Get("decompose"));
    if (!tree) { std::cerr << "no decompose tree\n"; return; }

    int n = tree->GetEntries();

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

    if (strcmp(stream, "tmix") == 0 && layer_lo < 1) layer_lo = 1;

    std::cout << "stream  layer  n_embd  ||dir||    gap       out_file\n";
    std::cout << "------  -----  ------  --------  --------  --------\n";

    for (int il = layer_lo; il <= layer_hi; ++il) {
        TString br_name = TString::Format("%s_L%d", stream, il);
        auto * br = tree->GetBranch(br_name.Data());
        if (!br) continue;

        int n_embd = br->GetLeaf(br_name.Data())->GetLen();

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

        std::vector<float> dir(n_embd);
        double norm = 0;
        for (int j = 0; j < n_embd; ++j) {
            dir[j] = (float)(mean_dang[j] - mean_safe[j]);
            norm += (double)dir[j] * dir[j];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < n_embd; ++j) dir[j] /= (float)norm;

        double proj_safe = 0, proj_dang = 0;
        for (int j = 0; j < n_embd; ++j) {
            proj_safe += mean_safe[j] * dir[j];
            proj_dang += mean_dang[j] * dir[j];
        }

        TString out_file = TString::Format("%s/%s_L%d.bin", out_dir, stream, il);
        std::ofstream ofs(out_file.Data(), std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(dir.data()), n_embd * sizeof(float));
        ofs.close();

        std::cout << TString::Format("%-6s  L%-4d  %5d   %8.3f  %8.3f  %s\n",
                     stream, il, n_embd, norm, proj_dang - proj_safe, out_file.Data());
    }

    fin->Close();
    std::cout << "\ndone.\n";
}
