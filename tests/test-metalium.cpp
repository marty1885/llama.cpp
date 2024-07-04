// This file is like test-backend-ops.cpp but we expect _everything_ to be supported by the Metalium backend.
// Also tests for edge cases in Metalium. (ex: Metalium/TTNN nativly uses 32x32 matrices as it's smallest unit)

// some code stolen from test-backend-ops.cpp
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <functional>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <iostream>

#if !defined (GGML_USE_METALIUM)
    #error "This file should only be compiled with Metalium backend enabled"
#endif

#include <ggml-metalium.h>

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    ggml_type_traits_t tt = ggml_internal_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt.to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ASSERT(false);
                    }
                }
            }
        }
    }

    return tv;
}

static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static bool isinf_or_max(float f) {
    return std::isinf(f) || f == std::numeric_limits<float>::max() || f == -std::numeric_limits<float>::max();
}

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    static std::default_random_engine generator(42);
    std::uniform_real_distribution<float> distribution(min, max);
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    for (size_t i = 0; i < size; i++) {
        data[i] = distribution(generator);
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, size));
        std::vector<float> imatrix(tensor->ne[0], 1.0f); // dummy importance matrix
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);
        GGML_ASSERT(ggml_validate_row_data(tensor->type, dataq.data(), dataq.size()));
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ASSERT(false);
    }
}

void initialize_tensors(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        init_tensor_uniform(t);
    }
}

enum class TestResult {
    OK,
    FAIL,
    NOT_SUPPORTED
};

struct test_case
{
    test_case(std::string name, std::function<ggml_tensor* (ggml_context*)> build_graph, const std::function<double(const float*, const float*, size_t n)>& loss = nmse)
        : name(std::move(name)), build_graph(std::move(build_graph)), loss(loss) {}
    std::string name;
    float max_err = 1e-4;
    std::function<double(const float*, const float*, size_t n)> loss;
    std::function<ggml_tensor* (ggml_context*)> build_graph;

    static const int sentinel_size = 1024;
    std::vector<ggml_tensor *> sentinels;
    ggml_cgraph * gf = nullptr;

    void add_sentinel(ggml_context * ctx) {
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    TestResult eval(ggml_backend_t backend1, ggml_backend_t backend2) {
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        printf("  %s (%s): ", name.c_str(), ggml_op_desc(out));
        fflush(stdout);

        // check if the backends support the ops
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("\033[1;33mNOT_SUPPORTED\033[0m by [%s]. Rejected OP: %s\n", ggml_backend_name(backend), ggml_op_desc(t));
                    ggml_free(ctx);
                    return TestResult::NOT_SUPPORTED;
                }
            }
        }
        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return TestResult::FAIL;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            gf->nodes[gf->n_nodes++] = sentinel;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
            std::function<double(const float*, const float*, size_t n)> loss;
        };

        callback_userdata ud {
            true,
            max_err,
            backend1,
            backend2,
            loss
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = ud->loss(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] loss = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);

        if (!cmp_ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return TestResult::OK;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return TestResult::FAIL;
    }
};

std::unique_ptr<test_case> make_test(const std::function<ggml_tensor* (ggml_context*)> & build_graph, std::string name, float max_err = 1e-4) {
    std::unique_ptr<test_case> tc = std::make_unique<test_case>(std::move(name), build_graph);
    tc->max_err = max_err;
    return tc;
}

std::string type_name(ggml_type type)
{
    return ggml_internal_get_type_traits(type).type_name;
}

int main()
{
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t metalium = ggml_backend_metalium_init();

    std::vector<std::unique_ptr<test_case>> tests;

    const ggml_unary_op supported_unary_ops[] = {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        // GGML_UNARY_OP_STEP, // Not supported by Metalium
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
    };

    // TODO: Add more types
    const ggml_type supported_types[] = {
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_BF16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q4_0
    };

    for(auto type : supported_types) {
        for(auto op : supported_unary_ops) {
            tests.push_back(make_test([op](ggml_context* ctx) {
                ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
                return ggml_unary(ctx, a, op);
            }, "Basic activation function for " + type_name(type), 1e-2));
        }
    }

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_cont(ctx, a);
        return b;
    }, "CONT on real tesnor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 64, 64, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "No-op view"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View into 2D matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, a->nb[1], ggml_type_size(GGML_TYPE_F32));
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View into 2D matrix with offset"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 48, 1, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "1D view into 2D matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 30, 28, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "Rectangular view into 2D matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        return ggml_transpose(ctx, a);
    }, "transpose 2D square matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 28);
        return ggml_transpose(ctx, a);
    }, "transpose 2D rectangular matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_transpose(ctx, a);
    }, "transpose 3D square matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_cont(ctx, ggml_transpose(ctx, a));
    }, "transpose 3D square matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 256, 4, 4, 4);
        ggml_cpy(ctx, a, b);
        return b;
    }, "4D tensor copy"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
        return ggml_view_2d(ctx, ggml_transpose(ctx, a), 4, 12, 4 * 4, 0);
    }, "View of transposed 4D tensor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 64, 4, 1);
        return ggml_reshape_4d(ctx, a, 32, 128, 4, 1);
    }, "Reshape to tile aligned tensor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 32, 1, 1);
        return ggml_reshape_4d(ctx, a, 16, 32, 2, 1);
    }, "Reshape to non tile aligned tensor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup_tensor(ctx, a);
    }, "Tensor duplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup_tensor(ctx, ggml_view_tensor(ctx, a));
    }, "Tensor duplication via view"));

    // (Basics of) what we need to get KV cache working
    // TODO: Map GGML operations into TTNN nlp_kv_cache_load_slice and update_cache_multi_core
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 24);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        return ggml_set_2d(ctx, a, b, b->nb[1], 0);
    }, "Set row of 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 24);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        return ggml_set_2d(ctx, a, b, b->nb[1], a->nb[1]);
    }, "Set row of 2D matrix with offset"));

    // Matrix multiplication
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        return ggml_mul_mat(ctx, a, b);
    }, "2D matrix multiplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 128);
        return ggml_mul_mat(ctx, a, b);
    }, "2D matrix multiplication (result non square)"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 38, 64);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 38, 72);
        return ggml_mul_mat(ctx, a, b);
    }, "2D matrix multiplication (result non square, non tile aligned)"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 10);
        ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 10);
        return ggml_mul_mat(ctx, a, b);
    }, "3D matrix multiplication"));
    // TODO: TTNN seems to not support the style of broadcasting GGML wants
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 20);
    //     ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 10);
    //     return ggml_mul_mat(ctx, a, b);
    // }, "3D matrix multiplication (broadcast)"));

    // Misc
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_clamp(ctx, a, -0.1, 0.25);
    }, "Clamp"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_scale(ctx, a, 2.0);
    }, "Scale"));
    // ???? This should not have worked since I haven't implemented inplace operations
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
    //     return ggml_scale_inplace(ctx, a, 2.0);
    // }, "Scale in place"));

    // more complex tests
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 18);
        ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
        ggml_tensor* b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor* w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 48);
        ggml_tensor* b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 48);

        ggml_tensor* h1 = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, x), b1));
        ggml_tensor* h2 = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, h1), b2));

        return h2;
    }, "Multi layer perceptron"));

    size_t total_tests = 0;
    size_t passed_tests = 0;
    size_t not_supported = 0;
    for(auto& test : tests) {
        TestResult res = test->eval(metalium, cpu);

        total_tests++;
        if (res == TestResult::OK) {
            passed_tests++;
        } else if (res == TestResult::NOT_SUPPORTED) {
            not_supported++;
        }
    }

    double passed_ratio = (double)passed_tests / total_tests;
    std::cout << "\nStats for Metalium backend: " << (total_tests != passed_tests ? "\033[1;31mFAIL\033[0m\n" : "\033[1;32mOK\033[0m\n") 
        << "  Test status: " << passed_tests << " / " << total_tests << " passed\n"
        << "  Failed: " << total_tests - passed_tests << " (" << (1.0 - passed_ratio) * 100 << "%)\n"
        << "  Not supported: " << not_supported << "\n";
    
    if(total_tests != passed_tests) {
        std::cout << "Some tests failed\n";
        return 1;
    }

    ggml_backend_free(metalium);
    ggml_backend_free(cpu);
}