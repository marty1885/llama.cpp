// probe_tmva.C — TMVA probes (Fisher + BDT + SVM) on PCA-reduced features.
//
// Uses dual PCA trick (500×500 Gram matrix instead of 2560×2560 covariance)
// to reduce dimensions, then runs TMVA classification.
//
// Usage:
//   root -l -b -q 'probe_tmva.C+("build/decompose_500.root")'
//   root -l -b -q 'probe_tmva.C+("file.root", "tmix", 1, 31, 50)'

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"

#include <algorithm>
#include <iostream>
#include <vector>

// Project data[n×d] onto top n_pca PCs via Gram matrix eigendecomposition.
// Returns projected[n][n_pca] and variance explained ratio.
static void dual_pca_project(const std::vector<float> & data, int n, int d,
                             int n_pca,
                             std::vector<std::vector<float>> & projected,
                             double & var_explained) {
    // mean
    std::vector<double> mean(d, 0.0);
    for (int i = 0; i < n; ++i) {
        const float * row = &data[i * d];
        for (int j = 0; j < d; ++j) mean[j] += row[j];
    }
    for (int j = 0; j < d; ++j) mean[j] /= n;

    // Gram matrix
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

    TMatrixDSymEigen eig(G);
    const TVectorD & eigval = eig.GetEigenValues();
    const TMatrixD & eigvec = eig.GetEigenVectors();

    double total_var = 0;
    for (int i = 0; i < n; ++i) total_var += eigval[i];

    int actual = std::min(n_pca, n);
    // eigenvalues are in DESCENDING order in ROOT's TMatrixDSymEigen
    var_explained = 0;
    for (int k = 0; k < actual; ++k)
        var_explained += eigval[k];
    var_explained = (total_var > 0) ? var_explained / total_var : 0;

    // projections = eigenvectors of Gram, scaled by sqrt(eigenvalue)
    projected.resize(n, std::vector<float>(actual));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < actual; ++k) {
            int col = k;  // descending order: 0 = largest
            double scale = (eigval[col] > 0) ? std::sqrt(eigval[col]) : 0;
            projected[i][k] = (float)(eigvec(i, col) * scale);
        }
    }
}

void probe_tmva(
    const char * file     = "decompose_500.root",
    const char * stream   = "tmix",
    int          layer_lo = 1,
    int          layer_hi = 31,
    int          n_pca    = 20)
{
    auto * fin = TFile::Open(file);
    if (!fin || fin->IsZombie()) { std::cerr << "cannot open " << file << "\n"; return; }
    auto * tree = dynamic_cast<TTree *>(fin->Get("decompose"));
    if (!tree) { std::cerr << "no decompose tree\n"; return; }

    int n = tree->GetEntries();

    int n_embd = 2560;
    {
        TString test_br = TString::Format("%s_L%d", stream,
                          (strcmp(stream, "tmix") == 0) ? 1 : 0);
        auto * br = tree->GetBranch(test_br.Data());
        if (br && br->GetLeaf(test_br.Data()))
            n_embd = br->GetLeaf(test_br.Data())->GetLen();
    }

    std::vector<int> labels(n);
    {
        Int_t buf;
        tree->SetBranchAddress("label", &buf);
        for (int i = 0; i < n; ++i) { tree->GetEntry(i); labels[i] = buf; }
        tree->ResetBranchAddresses();
    }

    int n_safe = std::count(labels.begin(), labels.end(), 0);
    int n_dang = std::count(labels.begin(), labels.end(), 1);
    std::cout << "entries=" << n << "  safe=" << n_safe
              << "  dang=" << n_dang << "  n_embd=" << n_embd
              << "  n_pca=" << n_pca << "\n\n";

    if (strcmp(stream, "tmix") == 0 && layer_lo < 1) layer_lo = 1;

    TString tmva_out = TString::Format("probe_tmva_%s.root", stream);
    auto * fout = TFile::Open(tmva_out, "RECREATE");

    struct LayerResult { int layer; double bdt; double svm; double var; };
    std::vector<LayerResult> results;

    std::vector<float> data(n * n_embd);
    std::vector<float> branch_buf(n_embd);

    std::cerr << "running TMVA (output is noisy, clean table printed at end)...\n";

    for (int il = layer_lo; il <= layer_hi; ++il) {
        TString br_name = TString::Format("%s_L%d", stream, il);
        if (!tree->GetBranch(br_name.Data())) continue;

        std::cerr << "L" << il << "..." << std::flush;

        // load
        tree->SetBranchAddress(br_name.Data(), branch_buf.data());
        for (int i = 0; i < n; ++i) {
            tree->GetEntry(i);
            std::copy(branch_buf.begin(), branch_buf.end(), data.begin() + i * n_embd);
        }
        tree->ResetBranchAddresses();

        // PCA reduce
        std::vector<std::vector<float>> projected;
        double var_expl;
        dual_pca_project(data, n, n_embd, n_pca, projected, var_expl);
        int actual = projected[0].size();

        // build separate signal / background trees
        TString sig_name = TString::Format("sig_%s_L%d", stream, il);
        TString bkg_name = TString::Format("bkg_%s_L%d", stream, il);
        auto * sig_tree = new TTree(sig_name, sig_name);
        auto * bkg_tree = new TTree(bkg_name, bkg_name);

        std::vector<Float_t> pc_buf(actual);

        for (int k = 0; k < actual; ++k) {
            TString vname = TString::Format("pc%d", k);
            sig_tree->Branch(vname, &pc_buf[k], vname + "/F");
            bkg_tree->Branch(vname, &pc_buf[k], vname + "/F");
        }

        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < actual; ++k) pc_buf[k] = projected[i][k];
            if (labels[i] == 1) sig_tree->Fill();
            else                bkg_tree->Fill();
        }

        // TMVA
        TString loader_name = TString::Format("%s_L%d", stream, il);
        auto * loader = new TMVA::DataLoader(loader_name.Data());
        for (int k = 0; k < actual; ++k)
            loader->AddVariable(TString::Format("pc%d", k), 'F');

        loader->AddSignalTree(sig_tree, 1.0);
        loader->AddBackgroundTree(bkg_tree, 1.0);
        loader->PrepareTrainingAndTestTree(
            TCut(""), TCut(""),
            "nTrain_Signal=0:nTrain_Background=0:"
            "SplitMode=Random:SplitSeed=42:NormMode=NumEvents:!V");

        // TMVA is noisy, results collected and printed at the end

        auto * factory = new TMVA::Factory(TString::Format("L%d", il), fout,
            "!V:Silent:!Color:!DrawProgressBar:AnalysisType=Classification");

        factory->BookMethod(loader, TMVA::Types::kBDT, "BDT",
            "NTrees=200:MaxDepth=3:BoostType=AdaBoost:nCuts=20:!V");
        factory->BookMethod(loader, TMVA::Types::kSVM, "SVM",
            "Kernel=RBF:Gamma=0.02:C=1.0:!V");

        factory->TrainAllMethods();
        factory->TestAllMethods();
        factory->EvaluateAllMethods();

        auto get_auc = [&](const char * method) -> double {
            TString path = TString::Format("%s/Method_%s/%s/MVA_%s_rejBvsS",
                                           loader_name.Data(), method, method, method);
            auto * h = dynamic_cast<TH1 *>(fout->Get(path));
            if (!h) return -1;
            return h->Integral() / h->GetNbinsX();
        };

        double auc_b = get_auc("BDT");
        double auc_s = get_auc("SVM");

        results.push_back({il, auc_b, auc_s, var_expl});
        std::cerr << " done\n";

        delete factory;
        delete loader;
        delete sig_tree;
        delete bkg_tree;
    }

    fout->Close();
    fin->Close();

    // ── clean summary table to stdout ───────────────────────────────────────
    std::cout << "\n";
    std::cout << "=== TMVA probe: " << stream << " (PCA " << n_pca << ") ===\n";
    std::cout << "layer  BDT_AUC  SVM_AUC  var_expl\n";
    std::cout << "-----  -------  -------  --------\n";
    for (const auto & r : results) {
        std::cout << TString::Format("L%-4d  %.3f    %.3f    %.1f%%\n",
                                     r.layer, r.bdt, r.svm, r.var * 100);
    }
    std::cout << "\nTMVA output: " << tmva_out << "\n";
}
