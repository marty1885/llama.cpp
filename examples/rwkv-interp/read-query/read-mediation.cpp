#define main collector_main
#include "../collect.cpp"
#undef main

struct mediation_trace {
    trace base;
    int source_layer = -1, mediator_layer = -1, mediator_mode = 0;
    int r_count = 0, source_hits = 0, mediator_hits = 0;
    std::vector<float> delta;
    const std::vector<float> * reference = nullptr;
    mediation_trace(int l, int e, int h) : base(l, e, h) {}
    size_t offset(int layer, int field) const { return (size_t(layer)*11 + field)*base.embd; }
    void clear() { base.clear(); r_count = source_hits = mediator_hits = 0; }
    static bool callback(ggml_tensor * t, bool ask, void * user) noexcept {
        auto & s = *static_cast<mediation_trace *>(user);
        bool r_op = t->op == GGML_OP_MUL_MAT && t->src[0] && std::string(t->src[0]->name).find(".time_mix_receptance.weight") != std::string::npos;
        if (ask) return r_op || trace::callback(t, true, &s.base);
        try {
            if (r_op) {
                int layer = s.r_count++;
                if (layer == s.mediator_layer && s.mediator_mode == 1) {
                    check(s.reference && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nbytes(t) == size_t(s.base.embd)*sizeof(float), "r patch shape mismatch");
                    ggml_backend_tensor_set(t, s.reference->data() + s.offset(layer, 0), 0, size_t(s.base.embd)*sizeof(float));
                    ++s.mediator_hits;
                }
                return true;
            }
            if (!trace::callback(t, false, &s.base)) return false;
            if (t->op != GGML_OP_RWKV_WKV7) return true;
            int layer = s.base.cursor - 1;
            int e = s.base.embd, d = s.base.head;
            auto at = s.offset(layer, 6);
            if (layer == s.mediator_layer && s.mediator_mode == 1) {
                check(std::equal(s.reference->begin() + s.offset(layer, 0), s.reference->begin() + s.offset(layer, 0) + e, s.base.data.begin() + s.offset(layer, 0)), "r restoration readback failed");
            }
            if (layer != s.source_layer && !(layer == s.mediator_layer && s.mediator_mode == 2)) return true;
            std::vector<float> value(s.base.data.begin() + at, s.base.data.begin() + at + e);
            std::vector<float> state(size_t(e)*d), after(state.size());
            ggml_backend_tensor_get(t, state.data(), size_t(e)*sizeof(float), state.size()*sizeof(float));
            if (layer == s.source_layer) {
                check(s.delta.size() == size_t(e), "source patch shape mismatch");
                for (int i = 0; i < e; ++i) value[i] += s.delta[i];
                ++s.source_hits;
            }
            if (layer == s.mediator_layer && s.mediator_mode == 2) {
                check(s.reference != nullptr, "missing mediator reference");
                for (int h = 0; h < e/d; ++h) {
                    double kr = 0;
                    for (int j = 0; j < d; ++j) {
                        double dr = (*s.reference)[s.offset(layer, 0) + h*d + j] - s.base.data[s.offset(layer, 0) + h*d + j];
                        kr += s.base.data[s.offset(layer, 2) + h*d + j]*dr;
                    }
                    for (int i = 0; i < d; ++i) {
                        double change = -s.base.data[s.offset(layer, 3) + h*d + i]*kr;
                        for (int j = 0; j < d; ++j) {
                            double dr = (*s.reference)[s.offset(layer, 0) + h*d + j] - s.base.data[s.offset(layer, 0) + h*d + j];
                            change += state[(size_t(h)*d + i)*d + j]*dr;
                        }
                        value[h*d + i] += float(change);
                    }
                }
                ++s.mediator_hits;
            }
            for (float x : value) check(std::isfinite(x), "nonfinite read intervention");
            ggml_backend_tensor_set(t, value.data(), 0, size_t(e)*sizeof(float));
            ggml_backend_tensor_get(t, after.data(), size_t(e)*sizeof(float), state.size()*sizeof(float));
            check(state == after, "read patch changed recurrent state");
            ggml_backend_tensor_get(t, s.base.data.data() + at, 0, size_t(e)*sizeof(float));
            check(std::equal(value.begin(), value.end(), s.base.data.begin() + at), "read patch readback failed");
            return true;
        } catch (...) { s.base.error = std::current_exception(); return false; }
    }
};

struct mediation_run {
    std::vector<llama_token> input;
    rwkv::checkpoint before;
    std::vector<std::vector<float>> vectors;
    std::vector<float> logits;
};

static std::vector<float> current_read(const std::vector<float> & data, int layer, int e, int d) {
    std::vector<float> result(e);
    auto at = size_t(layer)*11*e;
    for (int h = 0; h < e/d; ++h) {
        double kr = 0;
        for (int j = 0; j < d; ++j) kr += double(data[at + 2*e + h*d + j])*data[at + h*d + j];
        for (int i = 0; i < d; ++i) result[h*d + i] = float(data[at + 3*e + h*d + i]*kr);
    }
    return result;
}

int main(int argc, char ** argv) {
    try {
        check(argc == 4, "usage: read-mediation MODEL PLAN OUTPUT");
        fs::path root(argv[3]); check(!fs::exists(root), "output exists"); fs::create_directories(root);
        load_backends();
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("Vulkan0"), nullptr};
        check(devices[0] != nullptr, "Vulkan0 unavailable");
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 99; mp.devices = devices;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        check(bool(model), "model load failed");
        int l = llama_model_n_layer(model.get()), e = llama_model_n_embd(model.get());
        char head_text[64]; check(llama_model_meta_val_str(model.get(), "rwkv7.wkv.head_size", head_text, sizeof(head_text)) > 0, "RWKV7 required");
        int d = std::stoi(head_text); check(d > 0 && e%d == 0, "invalid head size");
        mediation_trace tr(l, e, d);
        auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = 8; cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.cb_eval = mediation_trace::callback; cp.cb_eval_user_data = &tr;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        check(bool(ctx), "context creation failed"); rwkv::session session(ctx.get());
        const auto * vocab = llama_model_get_vocab(model.get());
        auto yellow = tokenize(vocab, " yellow"), blue = tokenize(vocab, " blue");
        check(yellow.size() == 1 && blue.size() == 1, "single-token colors required");
        auto margin = [&](const std::vector<float> & logits) { return double(logits.at(blue[0])) - logits.at(yellow[0]); };
        auto step = [&](llama_token token) {
            tr.clear(); rwkv::session::request request{0, {token}};
            auto logits = session.prefill(std::span(&request, 1)).front().logits;
            tr.base.validate(); check(tr.r_count == l, "missing r producer callbacks");
            for (float x : tr.base.data) check(std::isfinite(x), "nonfinite activation");
            for (float x : logits) check(std::isfinite(x), "nonfinite logits");
            return logits;
        };
        std::ifstream plan(argv[2]); check(bool(plan), "plan open failed");
        std::vector<std::array<std::string, 5>> rows; std::string line;
        while (std::getline(plan, line)) {
            std::array<std::string, 5> row; std::istringstream stream(line);
            for (auto & field : row) std::getline(stream, field, '\t');
            for (size_t p = 0; (p = row[3].find("\\n", p)) != std::string::npos;) row[3].replace(p, 2, "\n");
            rows.push_back(row);
        }
        check(plan.eof() && rows.size() == 24, "expected eight crossed stories");
        std::vector<mediation_run> baseline(16);
        for (int s = 0; s < 8; ++s) {
            check(rows[s*3][0] == "s" + std::to_string(s), "unexpected story order");
            check(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1), "reset failed");
            for (auto token : tokenize(vocab, rows[s*3][3])) step(token);
            auto checkpoint = session.capture(0);
            for (int q = 0; q < 2; ++q) {
                auto & run = baseline[s*2 + q]; session.restore(0, checkpoint);
                run.input = tokenize(vocab, rows[s*3 + q + 1][3]); check(run.input.size() == 12, "expected twelve query tokens");
                for (int t = 0; t < 12; ++t) {
                    if (t == 9) run.before = session.capture(0);
                    run.logits = step(run.input[t]); run.vectors.push_back(tr.base.data);
                }
            }
            std::cout << "baseline " << s << std::endl;
        }
        auto donor_index = [](int i) { return ((i/2)^2)*2 + (1 - i%2); };
        for (int i = 0; i < 16; ++i) {
            check(baseline[i].input == baseline[donor_index(i)].input, "literal queries differ");
            check(margin(baseline[i].logits)*margin(baseline[donor_index(i)].logits) < 0, "baseline answers do not differ");
            int s = i/2, q = i%2; bool is_blue = (q == 1) != (s >= 4);
            check((margin(baseline[i].logits) > 0) == is_blue, "incorrect baseline answer");
        }
        auto make_delta = [&](int i, int layer, int part) {
            const auto & a = baseline[i].vectors[9]; const auto & b = baseline[donor_index(i)].vectors[9];
            auto ca = current_read(a, layer, e, d), cb = current_read(b, layer, e, d);
            std::vector<float> delta(e, 0); auto at = tr.offset(layer, 6);
            for (int j = 0; j < e; ++j) {
                if (part == 0) delta[j] = (b[at + j] - cb[j]) - (a[at + j] - ca[j]);
                if (part == 1) delta[j] = cb[j] - ca[j];
            }
            return delta;
        };
        auto run_case = [&](int i, int source, int part, int mediator, int mode, const mediation_run * reference) {
            mediation_run result; const auto & base = baseline[i]; session.restore(0, base.before);
            tr.delta = source >= 0 ? make_delta(i, source, part) : std::vector<float>();
            for (int t = 9; t < 12; ++t) {
                tr.source_layer = t == 9 ? source : -1; tr.mediator_layer = mediator; tr.mediator_mode = mode;
                tr.reference = reference ? &reference->vectors[t - (reference == &base ? 0 : 9)] : nullptr;
                result.logits = step(base.input[t]); result.vectors.push_back(tr.base.data);
                check(tr.source_hits == (t == 9 && source >= 0 ? 1 : 0), "source patch count mismatch");
                check(tr.mediator_hits == (mediator >= 0 ? 1 : 0), "mediator patch count mismatch");
                if ((source < 0 || part == 2) && (mode == 0 || reference == &base)) check(tr.base.data == base.vectors[t], "self control changes activations");
                if (source >= 0 && mediator < 0) for (int layer = 0; layer <= source; ++layer) for (int field = 0; field < 11; ++field) {
                    if (layer == source && field >= 6) continue;
                    auto at = tr.offset(layer, field);
                    check(std::equal(base.vectors[t].begin() + at, base.vectors[t].begin() + at + e, tr.base.data.begin() + at), "source patch changes upstream operands");
                }
            }
            tr.source_layer = tr.mediator_layer = -1; tr.mediator_mode = 0; tr.reference = nullptr;
            return result;
        };
        std::ofstream scores(root/"scores.tsv"), changes(root/"read-changes.tsv");
        scores << std::setprecision(17) << "case\tstage\tsource_layer\tmediator_layer\tbaseline_margin\tdonor_margin\tmargin\trecovery\n";
        changes << std::setprecision(17) << "case\tsource_layer\ttoken\tlayer\tfield\trelative_change\tdonor_projection\n";
        auto save = [&](int i, const char * stage, int source, int mediator, const mediation_run & run) {
            double a = margin(baseline[i].logits), b = margin(baseline[donor_index(i)].logits), c = margin(run.logits);
            scores << i << '\t' << stage << '\t' << source << '\t' << mediator << '\t' << a << '\t' << b << '\t' << c << '\t' << (c - a)/(b - a) << '\n';
            return (c - a)/(b - a);
        };
        std::vector<double> selection(l, 0);
        for (int layer = 0; layer < l; ++layer) {
            for (int i = 0; i < 16; ++i) {
                auto changed = run_case(i, layer, 0, -1, 0, nullptr);
                double recovery = save(i, "old_read", layer, -1, changed);
                if (i < 8) selection[layer] += recovery/8;
                auto self = run_case(i, layer, 2, -1, 0, nullptr);
                check(self.logits == baseline[i].logits, "self control changes logits"); save(i, "self", layer, -1, self);
            }
            scores.flush(); std::cout << "source layer " << layer << std::endl;
        }
        int source = int(std::max_element(selection.begin(), selection.end() - 1) - selection.begin());
        std::ofstream selected(root/"selected.tsv"); selected << "source_layer\t" << source << "\ntraining_mean_recovery\t" << selection[source] << '\n'; selected.close();
        for (int i = 0; i < 16; ++i) {
            auto changed = run_case(i, source, 0, -1, 0, nullptr);
            save(i, "source_repeat", source, -1, changed);
            auto immediate = run_case(i, source, 1, -1, 0, nullptr); save(i, "current_read", source, -1, immediate);
            for (int t = 9; t < 12; ++t) for (int layer = 0; layer < l; ++layer) for (int field : {0, 6}) {
                auto at = tr.offset(layer, field); double dd = 0, bb = 0, dot = 0, nn = 0;
                for (int j = 0; j < e; ++j) {
                    double a = baseline[i].vectors[t][at + j], dx = changed.vectors[t - 9][at + j] - a;
                    double dy = baseline[donor_index(i)].vectors[t][at + j] - a;
                    dd += dx*dx; bb += a*a; dot += dx*dy; nn += dy*dy;
                }
                changes << i << '\t' << source << '\t' << t << '\t' << layer << '\t' << field << '\t' << std::sqrt(dd/std::max(bb, 1e-30)) << '\t' << dot/std::max(nn, 1e-30) << '\n';
            }
            for (int mediator = source + 1; mediator < l; ++mediator) {
                auto full = run_case(i, source, 0, mediator, 1, &baseline[i]); save(i, "restore_r", source, mediator, full);
                auto old = run_case(i, source, 0, mediator, 2, &baseline[i]); save(i, "restore_old_query", source, mediator, old);
                auto self = run_case(i, -1, 2, mediator, 1, &baseline[i]);
                check(self.logits == baseline[i].logits, "r self control changes logits"); save(i, "r_self", source, mediator, self);
                auto replay = run_case(i, source, 0, mediator, 1, &changed);
                check(replay.logits == changed.logits && replay.vectors == changed.vectors, "patched r replay changes result");
                save(i, "r_replay", source, mediator, replay);
            }
            scores.flush(); changes.flush(); check(bool(scores) && bool(changes), "output write failed");
            std::cout << "mediation case " << i << std::endl;
        }
        std::ofstream done(root/"DONE"); done << "complete\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
