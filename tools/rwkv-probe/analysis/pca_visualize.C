// pca_visualize.C — PCA scatter plots of decomposed residual stream.
//
// Projects each layer's vectors onto PC1 vs PC2, colored by label.
// Uses the dual PCA trick (Gram matrix) since n_samples << n_embd.
//
// Usage:
//   root -l -b -q 'pca_visualize.C+("build/decompose_500.root")'
//   root -l -b -q 'pca_visualize.C+("file.root", "tmix", 1, 31, "tmix_pca.pdf")'

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TCanvas.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TVectorD.h"

#include "TLeaf.h"

#include <iostream>
#include <vector>

// Dual PCA: eigendecompose the n×n Gram matrix instead of d×d covariance.
// Returns top-2 projections for each sample.
static void dual_pca(const std::vector<float> & data, int n, int d,
                     std::vector<double> & pc1, std::vector<double> & pc2,
                     double & var1_pct, double & var2_pct) {
    // compute mean
    std::vector<double> mean(d, 0.0);
    for (int i = 0; i < n; ++i) {
        const float * row = &data[i * d];
        for (int j = 0; j < d; ++j) mean[j] += row[j];
    }
    for (int j = 0; j < d; ++j) mean[j] /= n;

    // build Gram matrix G[i][k] = sum_j (x_ij - mean_j)(x_kj - mean_j)
    TMatrixDSym G(n);
    for (int i = 0; i < n; ++i) {
        const float * ri = &data[i * d];
        for (int k = i; k < n; ++k) {
            const float * rk = &data[k * d];
            double dot = 0;
            for (int j = 0; j < d; ++j)
                dot += ((double)ri[j] - mean[j]) * ((double)rk[j] - mean[j]);
            G(i, k) = dot;
            G(k, i) = dot;
        }
    }

    // eigendecompose (returns eigenvalues in ASCENDING order)
    TMatrixDSymEigen eig(G);
    const TVectorD & eigval = eig.GetEigenValues();
    const TMatrixD & eigvec = eig.GetEigenVectors();

    // eigenvalues are in DESCENDING order in ROOT's TMatrixDSymEigen
    int idx1 = 0;
    int idx2 = 1;

    double total_var = 0;
    for (int i = 0; i < n; ++i) total_var += eigval[i];
    var1_pct = (total_var > 0) ? eigval[idx1] / total_var * 100.0 : 0;
    var2_pct = (total_var > 0) ? eigval[idx2] / total_var * 100.0 : 0;

    // projections are just the eigenvectors of the Gram matrix (scaled)
    pc1.resize(n);
    pc2.resize(n);
    for (int i = 0; i < n; ++i) {
        pc1[i] = eigvec(i, idx1);
        pc2[i] = eigvec(i, idx2);
    }
}

void pca_visualize(
    const char * file       = "decompose_500.root",
    const char * stream     = "tmix",
    int          layer_lo   = 1,
    int          layer_hi   = 31,
    const char * out_pdf    = "pca_scatter.pdf")
{
    gStyle->SetOptStat(0);

    auto * fin = TFile::Open(file);
    if (!fin || fin->IsZombie()) { std::cerr << "cannot open " << file << "\n"; return; }
    auto * tree = dynamic_cast<TTree *>(fin->Get("decompose"));
    if (!tree) { std::cerr << "no decompose tree\n"; return; }

    int n = tree->GetEntries();

    // detect n_embd
    int n_embd = 2560;
    {
        TString test_br = TString::Format("%s_L%d", stream,
                          (strcmp(stream, "tmix") == 0) ? 1 : 0);
        auto * br = tree->GetBranch(test_br.Data());
        if (br && br->GetLeaf(test_br.Data()))
            n_embd = br->GetLeaf(test_br.Data())->GetLen();
    }

    // load labels
    std::vector<int> labels(n);
    {
        Int_t buf;
        tree->SetBranchAddress("label", &buf);
        for (int i = 0; i < n; ++i) { tree->GetEntry(i); labels[i] = buf; }
        tree->ResetBranchAddresses();
    }

    std::cout << "entries=" << n << "  n_embd=" << n_embd << "\n";

    if (strcmp(stream, "tmix") == 0 && layer_lo < 1) layer_lo = 1;

    auto * c = new TCanvas("c", "PCA", 800, 600);
    TString pdf_name(out_pdf);
    c->Print(pdf_name + "[");

    // pre-allocate branch buffer
    std::vector<float> data(n * n_embd);
    std::vector<float> branch_buf(n_embd);

    for (int il = layer_lo; il <= layer_hi; ++il) {
        TString br_name = TString::Format("%s_L%d", stream, il);
        if (!tree->GetBranch(br_name.Data())) continue;

        std::cout << "L" << il << "..." << std::flush;

        // load all rows for this branch
        tree->SetBranchAddress(br_name.Data(), branch_buf.data());
        for (int i = 0; i < n; ++i) {
            tree->GetEntry(i);
            std::copy(branch_buf.begin(), branch_buf.end(), data.begin() + i * n_embd);
        }
        tree->ResetBranchAddresses();

        // PCA via Gram matrix
        std::vector<double> pc1, pc2;
        double var1, var2;
        dual_pca(data, n, n_embd, pc1, pc2, var1, var2);

        // split by label
        std::vector<double> x_safe, y_safe, x_dang, y_dang;
        for (int i = 0; i < n; ++i) {
            if (labels[i] == 0) { x_safe.push_back(pc1[i]); y_safe.push_back(pc2[i]); }
            else                { x_dang.push_back(pc1[i]); y_dang.push_back(pc2[i]); }
        }

        auto * mg = new TMultiGraph();
        mg->SetTitle(TString::Format("%s L%d;PC1 (%.1f%%);PC2 (%.1f%%)",
                                     stream, il, var1, var2));

        auto * g_safe = new TGraph(x_safe.size(), x_safe.data(), y_safe.data());
        g_safe->SetMarkerStyle(20);
        g_safe->SetMarkerSize(0.5);
        g_safe->SetMarkerColor(kBlue);

        auto * g_dang = new TGraph(x_dang.size(), x_dang.data(), y_dang.data());
        g_dang->SetMarkerStyle(20);
        g_dang->SetMarkerSize(0.5);
        g_dang->SetMarkerColor(kRed);

        mg->Add(g_safe, "P");
        mg->Add(g_dang, "P");

        c->Clear();
        mg->Draw("A");

        auto * leg = new TLegend(0.72, 0.78, 0.88, 0.88);
        leg->AddEntry(g_safe, "safe", "p");
        leg->AddEntry(g_dang, "dangerous", "p");
        leg->SetBorderSize(0);
        leg->Draw();

        c->Update();
        c->Print(pdf_name);

        std::cout << " ok\n";
    }

    c->Print(pdf_name + "]");
    fin->Close();

    std::cout << "\nsaved: " << pdf_name << "\n";
}
