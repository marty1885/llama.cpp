#define main collector_main
#include "../collect.cpp"
#undef main

struct trajectory {
    std::vector<llama_token> input;
    std::vector<rwkv::checkpoint> before;
    std::vector<std::vector<float>> vectors;
    std::vector<float> logits;
};

static double log_probability(const std::vector<float> & logits, llama_token token) {
    double maximum = *std::max_element(logits.begin(), logits.end()), sum = 0;
    for (float value : logits) { check(std::isfinite(value), "nonfinite logits"); sum += std::exp(value - maximum); }
    return logits.at(token) - maximum - std::log(sum);
}

int main(int argc, char ** argv) {
    try {
        check(argc == 4, "usage: layer-swap MODEL PLAN OUTPUT");
        fs::path root(argv[3]);
        check(!fs::exists(root), "output exists");
        fs::create_directories(root);
        load_backends();
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("Vulkan0"), nullptr};
        check(devices[0] != nullptr, "Vulkan0 unavailable");
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 99; mp.devices = devices;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        check(bool(model), "model load failed");
        int layers = llama_model_n_layer(model.get()), embd = llama_model_n_embd(model.get());
        char head_text[64];
        check(llama_model_meta_val_str(model.get(), "rwkv7.wkv.head_size", head_text, sizeof(head_text)) > 0, "RWKV7 metadata missing");
        trace tr(layers, embd, std::stoi(head_text));
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = 8; cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.cb_eval = trace::callback; cp.cb_eval_user_data = &tr;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        check(bool(ctx), "context creation failed");
        rwkv::session session(ctx.get());
        const auto * vocab = llama_model_get_vocab(model.get());
        auto yellow = tokenize(vocab, " yellow"), blue = tokenize(vocab, " blue");
        check(yellow.size() == 1 && blue.size() == 1, "single-token candidates required");
        auto step = [&](llama_token token) {
            tr.clear();
            rwkv::session::request request{0, {token}};
            auto logits = session.prefill(std::span(&request, 1)).front().logits;
            tr.validate();
            for (float value : tr.data) check(std::isfinite(value), "nonfinite activation");
            return logits;
        };
        auto margin = [&](const std::vector<float> & logits) { return log_probability(logits, blue[0]) - log_probability(logits, yellow[0]); };
        std::ofstream scores(root/"scores.tsv"), changes(root/"later-reads.tsv");
        scores << std::setprecision(17) << "story\trecipient\tmode\tpatch_token\tpatch_layer\tbaseline_margin\tdonor_margin\tpatched_margin\trecovery\n";
        changes << std::setprecision(17) << "story\trecipient\tmode\tpatch_token\tpatch_layer\tobserved_token\tlayer\tfield\trelative_change\tdonor_direction_projection\n";
        std::ifstream plan(argv[2]); check(bool(plan), "plan open failed");
        std::vector<std::array<std::string, 5>> rows;
        std::string line;
        while (std::getline(plan, line)) {
            std::array<std::string, 5> row;
            std::istringstream stream(line);
            for (auto & field : row) std::getline(stream, field, '\t');
            for (size_t p = 0; (p = row[3].find("\\n", p)) != std::string::npos;) row[3].replace(p, 2, "\n");
            rows.push_back(row);
        }
        check(plan.eof() && rows.size()%3 == 0, "invalid plan");
        for (size_t index = 0; index < rows.size(); index += 3) {
            const auto & story = rows[index];
            check(story[2] == "story", "story row expected");
            check(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, -1), "reset failed");
            for (auto token : tokenize(vocab, story[3])) step(token);
            const auto checkpoint = session.capture(0);
            std::array<trajectory, 2> baseline;
            for (int q = 0; q < 2; ++q) {
                check(rows[index + q + 1][1] == story[0], "query parent mismatch");
                session.restore(0, checkpoint);
                auto & t = baseline[q];
                t.input = tokenize(vocab, rows[index + q + 1][3]);
                check(t.input.size() == 12, "expected twelve-token query");
                for (int token = 0; token < 12; ++token) {
                    if (token == 9 || token == 11) t.before.push_back(session.capture(0));
                    t.logits = step(t.input[token]);
                    t.vectors.push_back(tr.data);
                }
            }
            for (int q = 0; q < 2; ++q) for (int slot = 0; slot < 2; ++slot) for (int layer = 0; layer < layers; ++layer) for (int control = 0; control < 2; ++control) {
                int token_start = slot == 0 ? 9 : 11;
                const auto & recipient = baseline[q];
                const auto & donor = baseline[control ? q : 1 - q];
                const char * mode = control ? "self" : "cross";
                session.restore(0, recipient.before[slot]);
                auto offset = (size_t(layer)*11 + 10)*embd;
                tr.patch.assign(donor.vectors[token_start].begin() + offset, donor.vectors[token_start].begin() + offset + embd);
                tr.patch_hits = 0;
                std::vector<float> logits;
                for (int token = token_start; token < 12; ++token) {
                    tr.patch_layer = token == token_start ? layer : -1;
                    logits = step(recipient.input[token]);
                    if (token == token_start) {
                        check(tr.patch_hits == 1, "patch did not execute exactly once");
                        check(std::equal(tr.patch.begin(), tr.patch.end(), tr.data.begin() + offset), "patch readback differs");
                    }
                    if (control) check(tr.data == recipient.vectors[token], "self patch changes activations");
                    for (int later = 0; later < layers; ++later) for (int field : {0, 6}) {
                        auto at = (size_t(later)*11 + field)*embd;
                        double delta_sq = 0, base_sq = 0, direction_sq = 0, projection = 0;
                        for (int coordinate = 0; coordinate < embd; ++coordinate) {
                            double base = recipient.vectors[token][at + coordinate];
                            double delta = tr.data[at + coordinate] - base;
                            double direction = baseline[1 - q].vectors[token][at + coordinate] - base;
                            delta_sq += delta*delta; base_sq += base*base;
                            direction_sq += direction*direction; projection += delta*direction;
                        }
                        if (token == token_start && later <= layer) check(delta_sq == 0, "patch changes an upstream read");
                        changes << story[0] << '\t' << q << '\t' << mode << '\t' << token_start << '\t' << layer << '\t' << token << '\t' << later << '\t' << (field == 0 ? "r" : "read") << '\t' << std::sqrt(delta_sq/std::max(base_sq, 1e-30)) << '\t' << (direction_sq > 0 ? projection/direction_sq : 0) << '\n';
                    }
                }
                tr.patch_layer = -1;
                if (control) check(logits == recipient.logits, "self patch changes logits");
                double base = margin(recipient.logits), other = margin(baseline[1 - q].logits), changed = margin(logits);
                check(std::abs(other - base) > 1e-8, "baseline margin gap too small");
                scores << story[0] << '\t' << q << '\t' << mode << '\t' << token_start << '\t' << layer << '\t' << base << '\t' << other << '\t' << changed << '\t' << (changed - base)/(other - base) << '\n';
            }
            scores.flush(); changes.flush();
            check(bool(scores) && bool(changes), "output write failed");
            std::cout << "DONE " << story[0] << std::endl;
        }
        std::ofstream done(root/"DONE"); done << "complete\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
