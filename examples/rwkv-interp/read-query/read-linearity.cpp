#include <random>
#define main collector_main
#include "../collect.cpp"
#undef main

// The source patch follows read-mediation.cpp and preserves the packed recurrent state.
struct mediation_trace {
    trace base;
    int source_layer = -1;
    int r_count = 0, source_hits = 0;
    std::vector<float> delta;
    mediation_trace(int l, int e, int h) : base(l, e, h) {}
    size_t offset(int layer, int field) const { return (size_t(layer)*11 + field)*base.embd; }
    void clear() { base.clear(); r_count = source_hits = 0; }
    static bool callback(ggml_tensor * t, bool ask, void * user) noexcept {
        auto & s = *static_cast<mediation_trace *>(user);
        bool r_op = t->op == GGML_OP_MUL_MAT && t->src[0] && std::string(t->src[0]->name).find(".time_mix_receptance.weight") != std::string::npos;
        if (ask) return r_op || trace::callback(t, true, &s.base);
        try {
            if (r_op) { ++s.r_count; return true; }
            if (!trace::callback(t, false, &s.base)) return false;
            if (t->op != GGML_OP_RWKV_WKV7 || s.base.cursor - 1 != s.source_layer) return true;
            int e = s.base.embd;
            auto at = s.offset(s.source_layer, 6);
            check(s.delta.size() == size_t(e), "source patch shape mismatch");
            std::vector<float> value(e), state(size_t(e)*s.base.head), after(state.size());
            ggml_backend_tensor_get(t, state.data(), size_t(e)*sizeof(float), state.size()*sizeof(float));
            for (int i = 0; i < e; ++i) {
                value[i] = s.base.data[at + i] + s.delta[i];
                check(std::isfinite(value[i]), "nonfinite read intervention");
            }
            ggml_backend_tensor_set(t, value.data(), 0, size_t(e)*sizeof(float));
            ggml_backend_tensor_get(t, after.data(), size_t(e)*sizeof(float), state.size()*sizeof(float));
            check(state == after, "read patch changed recurrent state");
            ggml_backend_tensor_get(t, s.base.data.data() + at, 0, size_t(e)*sizeof(float));
            check(std::equal(value.begin(), value.end(), s.base.data.begin() + at), "read patch readback failed");
            ++s.source_hits;
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
        check(argc == 4, "usage: read-linearity MODEL PLAN OUTPUT");
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
        const int source = 34;
        check(l == 61 && e == 4096, "expected experiment model shape");
        auto natural_delta = [&](int i, bool colors) {
            int donor = colors ? ((i/2)^4)*2 + i%2 : donor_index(i);
            check(baseline[i].input == baseline[donor].input, "direction questions differ");
            const auto & a = baseline[i].vectors[9]; const auto & b = baseline[donor].vectors[9];
            auto ca = current_read(a, source, e, d), cb = current_read(b, source, e, d);
            std::vector<float> delta(e); auto at = tr.offset(source, 6);
            for (int j = 0; j < e; ++j) delta[j] = (b[at + j] - cb[j]) - (a[at + j] - ca[j]);
            return delta;
        };
        auto norm = [](const std::vector<float> & v) {
            double sum = 0; for (float x : v) sum += double(x)*x; return std::sqrt(sum);
        };
        std::vector<std::vector<float>> fixed{natural_delta(0, false), natural_delta(0, true)};
        std::mt19937 rng(20261006); std::normal_distribution<float> normal;
        for (int k = 0; k < 2; ++k) {
            std::vector<float> v(e); for (float & x : v) x = normal(rng);
            double scale = norm(fixed[k])/norm(v); for (float & x : v) x *= scale;
            fixed.push_back(v);
        }
        std::ofstream scores(root/"scores.tsv"), vectors(root/"queries.f32", std::ios::binary), inputs(root/"directions.f32", std::ios::binary);
        scores << std::setprecision(17) << "case\tdirection\talpha\tmargin\tbaseline_margin\tdelta_norm\n";
        std::ofstream config(root/"config.txt");
        config << "source_layer=34\nsource_token=9\nquery_layers=36,48,60\nquery_tokens=9,10,11\nembd=4096\ndirections=city,color,fixed_city,fixed_color,random_city_norm,random_color_norm,city_plus_color,random_sum\nseed=20261006\n";
        auto save_queries = [&](const std::vector<float> & data) {
            for (int layer : {36, 48, 60}) {
                auto at = tr.offset(layer, 0);
                write(vectors, std::vector<float>(data.begin() + at, data.begin() + at + e));
            }
        };
        int count = 0;
        for (int i = 0; i < 16; ++i) {
            std::vector<std::vector<float>> directions{natural_delta(i, false), natural_delta(i, true)};
            directions.insert(directions.end(), fixed.begin(), fixed.end());
            for (auto pair : {std::pair{0, 1}, std::pair{4, 5}}) {
                std::vector<float> v(e);
                for (int j = 0; j < e; ++j) v[j] = directions[pair.first][j] + directions[pair.second][j];
                directions.push_back(v);
            }
            for (const auto & v : directions) write(inputs, v);
            auto run = [&](int direction, double alpha) {
                const auto & base = baseline[i]; session.restore(0, base.before);
                tr.delta.assign(e, 0);
                if (direction >= 0) for (int j = 0; j < e; ++j) tr.delta[j] = alpha*directions[direction][j];
                std::vector<float> logits;
                for (int t = 9; t < 12; ++t) {
                    tr.source_layer = t == 9 ? source : -1;
                    logits = step(base.input[t]);
                    check(tr.source_hits == (t == 9 ? 1 : 0), "patch count mismatch");
                    if (direction < 0) check(tr.base.data == base.vectors[t], "zero patch changes activations");
                    if (t == 9) for (int layer = 0; layer <= source; ++layer) for (int field = 0; field < 11; ++field) {
                        if (layer == source && field >= 6) continue;
                        auto at = tr.offset(layer, field);
                        check(std::equal(base.vectors[t].begin() + at, base.vectors[t].begin() + at + e, tr.base.data.begin() + at), "patch changes upstream operands");
                    }
                    save_queries(tr.base.data);
                }
                if (direction < 0) check(logits == base.logits, "zero patch changes logits");
                scores << i << '\t' << direction << '\t' << alpha << '\t' << margin(logits) << '\t' << margin(base.logits) << '\t' << norm(tr.delta) << '\n';
                ++count; tr.source_layer = -1;
            };
            run(-1, 0);
            for (int direction = 0; direction < 6; ++direction) {
                for (double alpha : {-1., -.5, -.1, -.03, .03, .1, .5, 1., 2.}) run(direction, alpha);
            }
            for (int direction = 6; direction < 8; ++direction) for (double alpha : {.1, .5, 1.}) run(direction, alpha);
            scores.flush(); vectors.flush(); inputs.flush();
            check(bool(scores) && bool(vectors) && bool(inputs), "output write failed");
            std::cout << "linearity case " << i << " runs " << count << std::endl;
        }
        check(count == 976, "run count mismatch");
        std::ofstream done(root/"DONE"); done << "complete\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
