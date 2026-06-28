 // This file is like test-backend-ops.cpp but we expect _everything_ to be supported by the Metalium backend.
// Also tests for edge cases in Metalium. (ex: Metalium/TTNN nativly uses 32x32 matrices as it's smallest unit)

// some code stolen from test-backend-ops.cpp
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <functional>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

#if !defined (GGML_USE_METALIUM)
    #error "This file should only be compiled with Metalium backend enabled"
#endif

#include <ggml-metalium.h>

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
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
                        tt->to_float(&buf[i], vq.data(), bs);
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

static double pcc(const float * a, const float * b, size_t n) {
    // Calculate the mean of x and y values
    double a_mean = 0.0;
    double b_mean = 0.0;

    for (size_t i = 0; i < n; i++) {
        a_mean += a[i];
        b_mean += b[i];
    }

    a_mean /= n;
    b_mean /= n;

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float x_diff = a[i] - a_mean;
        float y_diff = b[i] - b_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= n;
    x_stddev /= n;
    y_stddev /= n;

    // Calculate the correlation coefficient
    double correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

static bool isinf_or_max(float f) {
    return std::isinf(f) || f == std::numeric_limits<float>::max() || f == -std::numeric_limits<float>::max();
}

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    static std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(min, max);
    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    for (size_t i = 0; i < size; i++) {
        data[i] = distribution(generator);
    }

    if (tensor->type == GGML_TYPE_F32) {
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
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> datai32(size);
        std::uniform_int_distribution<int32_t> distribution_int32(0, 2048);
        for (size_t i = 0; i < size; i++) {
            datai32[i] = distribution_int32(generator);
        }
        ggml_backend_tensor_set(tensor, datai32.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ASSERT(false);
    }
}

static void initialize_tensors(ggml_context * ctx) {
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
        : name(std::move(name)), loss(loss), build_graph(std::move(build_graph)) {}
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
        if(out->op == GGML_OP_NONE) {
            printf("\033[1;31mTEST_ERROR\033[0m operator should not be NONE. Test is buggy\n");
            return TestResult::FAIL;
        }
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
            ggml_graph_add_node(gf, sentinel);
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
                printf("[%s / %s] loss = %.9f > %.9f ", ggml_op_desc(t1), t1->name, err, ud->max_err);
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

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, &out, 1);

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

static std::unique_ptr<test_case> make_test(const std::function<ggml_tensor* (ggml_context*)> & build_graph, std::string name, float max_err = 1e-4) {
    std::unique_ptr<test_case> tc = std::make_unique<test_case>(std::move(name), build_graph);
    tc->max_err = max_err;
    return tc;
}

static std::string type_name(ggml_type type)
{
    return ggml_get_type_traits(type)->type_name;
}

static void add_unittests(std::vector<std::unique_ptr<test_case>>& tests)
{
    const ggml_unary_op supported_unary_ops[] = {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP, // Not supported by Metalium
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,
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
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 96, 96);
        ggml_tensor* v = ggml_view_2d(ctx, a, 64, 64, a->nb[1], 0);
        return ggml_unary(ctx, v, GGML_UNARY_OP_ABS);
    }, "Activation of view"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* b = ggml_cont(ctx, a);
        return b;
    }, "CONT on real tesnor"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 64, 64);
        ggml_tensor* b = ggml_cont(ctx, a);
        return b;
    }, "CONT on integer tesnor"));

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
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_1d(ctx, a, 32*32, 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, 32 * sizeof(float), 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32*32);
        ggml_tensor* view = ggml_view_2d(ctx, a, 32, 32, 32 * sizeof(float), 0);
        ggml_tensor* transposed = ggml_transpose(ctx, view);
        ggml_tensor* b = ggml_cont(ctx, transposed);
        return b;
    }, "Transposed flat buffer into 2D matrix"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 3D tensor"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 32 * sizeof(float));
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View flat buffer into 3D tensor with offset"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32 * 32 * 64);
        ggml_tensor* view = ggml_view_3d(ctx, a, 32, 32, 32, 32 * sizeof(float), 32 * 32 * sizeof(float), 0);
        ggml_tensor* transposed = ggml_transpose(ctx, view);
        ggml_tensor* b = ggml_cont(ctx, transposed);
        return b;
    }, "View flat buffer into 3D tensor transposed"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* view = ggml_view_2d(ctx, a, 30, 30, a->nb[1], 0);
        ggml_tensor* b = ggml_cont(ctx, view);
        return b;
    }, "View into 2D matrix, non tile aligned"));

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
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        return ggml_transpose(ctx, a);
    }, "transpose 2D small matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_transpose(ctx, a);
    }, "transpose 3D square matrix"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 64, 64);
        return ggml_cont(ctx, ggml_transpose(ctx, a));
    }, "transpose 3D square matrix"));

    // Failing - need to support tensor copy
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 256, 4, 4, 4);
    //     return ggml_cpy(ctx, a, b);
    // }, "4D tensor copy"));

    // Failing - need to support sum
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 14, 2, 3);
    //     return ggml_sum(ctx, a);
    // }, "sum"));

    // Failing - need to support sum_rows
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 14, 2, 3);
    //     return ggml_sum_rows(ctx, a);
    // }, "sum rows"));

    // Failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 16, 1, 4);
    //     return ggml_cpy(ctx, a, b);
    // }, "Copy tensor into tensor of different shape"));

    // FIXME: This sould work but is failing
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 4, 4, 4);
    //     return ggml_view_2d(ctx, ggml_transpose(ctx, a), 4, 12, 4 * 4, 0);
    // }, "View of transposed 4D tensor"));

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
        return ggml_dup(ctx, a);
    }, "Tensor duplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_dup(ctx, ggml_view_tensor(ctx, a));
    }, "Tensor duplication via view"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        ggml_tensor* view = ggml_view_tensor(ctx, a);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
        return ggml_cpy(ctx, view, b);
    }, "Write via view"));
    // Not working yet. Need write support for views
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 24, 2, 1);
    //     ggml_tensor* view = ggml_view_2d(ctx, a, 8, 12, a->nb[1], 1);
    //     ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 12);
    //     return ggml_cpy(ctx, view, b);
    // }, "partial write via view"));
    // TODO: Expend this to attempt all permutations possible

    std::array<int, GGML_MAX_DIMS> permute_order;
    for(int i = 0;i<GGML_MAX_DIMS;i++) {
        permute_order[i] = i;
    }
    do {
        std::string name = "Permutation, order=[";
        for(int i = 0;i<GGML_MAX_DIMS;i++) {
            name += std::to_string(permute_order[i]) + " ";
        }
        name.pop_back();
        name += "]";
        tests.push_back(make_test([permute_order](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 32, 8, 4);
            return ggml_permute(ctx, a, permute_order[0], permute_order[1], permute_order[2], permute_order[3]);
        }, name));
    } while(std::next_permutation(permute_order.begin(), permute_order.end()));

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
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        return ggml_mul_mat(ctx, a, b);
    }, "4D matrix multiplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 1);
        ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 64, 1, 10);
        return ggml_mul_mat(ctx, a, b);
    }, "4D matrix multiplication with broadcast"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        return ggml_mul_mat(ctx, a, b);
    }, "matrix-vector multiplication"));
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 18);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24);
        return ggml_mul_mat(ctx, a, b);
    }, "matrix-vector multiplication non tile aligned"));

    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 64);
        ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
        return ggml_add(ctx, a, b);
    }, "Add broadcasted vector to matrix"));

    // TODO: TTNN does not support the style of broadcasting GGML wants
    // Failing
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
    tests.push_back(make_test([](ggml_context* ctx) {
        ggml_tensor* a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 38, 64, 3, 26);
        return ggml_scale_inplace(ctx, a, 1.5);
    }, "Scale in place"));
    // RoPE
    for(auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) { // Really a limitation of GGML's CPU implementation - we support more
        tests.push_back(make_test([type](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 2048, 16, 2);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            return ggml_rope(ctx, a, b, 128, GGML_ROPE_TYPE_NEOX);
        }, "RoPE NEOX " + std::string(ggml_type_name(type))));

        tests.push_back(make_test([type](ggml_context* ctx) {
            float freq_base = 20000.f;
            float freq_scale = 1.4245f;
            float attn_factor = 1.424500f;
            float ext_factor = 0.746500f;
            float beta_fast = 32.f;
            float beta_slow = 1.f;
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 2048, 16, 2);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            return ggml_rope_ext(ctx, a, b, NULL, 128, GGML_ROPE_TYPE_NEOX, 512, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }, "RoPE NEOX " + std::string(ggml_type_name(type)) + " with YaRN"));

        tests.push_back(make_test([type](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, type, 512, 32, 1);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            return ggml_rope(ctx, a, b, 512, GGML_ROPE_TYPE_NEOX);
        }, "RoPE NEOX in Gemma " + std::string(ggml_type_name(type))));

        tests.push_back(make_test([](ggml_context* ctx) {
            ggml_tensor* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 1);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            return ggml_rope(ctx, a, b, 32, GGML_ROPE_TYPE_NORMAL);
        }, "RoPE Normal " + std::string(ggml_type_name(GGML_TYPE_F32))));
    }

    // TODO: Need a way to inform the RNG
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     int n = 300*256;
    //     int m = 60;
    //     int r = 8;
    //     int be1 = 1;
    //     int be2 = 1;
    //     ggml_tensor * in = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n, m, be1, be2);
    //     ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
    //     ggml_tensor * out = ggml_get_rows(ctx, in, rows);
    //
    // // DITTO
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     ggml_tensor * dst = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 32, 200, 1, 1);

    //     ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 1, 1, 1);
    //     ggml_tensor * idx = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1);

    //     ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);

    //     return out;
    // }, "test MM", 1e-5));

    //     return out;
    // }, "Simple GET_ROWS", 1e-5));
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

    tests.push_back(make_test([](ggml_context* ctx) {
        // A smaller and stripped down version of the MLP Mixer model
        ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
        ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
        std::array<ggml_tensor*, 4> h;
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ggml_tensor* patch = ggml_view_2d(ctx, in, 32, 32, in->nb[1], 32 * y + x * 32);
                h[y * 2 + x] = ggml_relu(ctx, ggml_mul_mat(ctx, w1, patch));
            }
        }
        ggml_tensor* h1 = ggml_concat(ctx, h[0], h[1], 1);
        ggml_tensor* h2 = ggml_concat(ctx, h[2], h[3], 1);
        ggml_tensor* h_all = ggml_concat(ctx, h1, h2, 1);
        return ggml_transpose(ctx, h_all);
    }, "MLP mixer", 1e-3));
}

// Round-trip the row-folded host contract: a tensor named cache_s_l* is stored on device as
// [1, n_rows, dim/32, 32] (folded), and GGML still addresses it in canonical [dim, n_rows] bytes.
// Verify whole-tensor set/get AND a sub-range (the [head, head+cells) state-restore pattern).
// Values are small integers (exact in bf16), so the round trip must be byte-for-byte.
static bool test_row_fold_roundtrip(ggml_backend_t backend) {
    const int64_t dim    = 256;   // tile-aligned inner dim
    const int64_t n_rows = 4;
    const size_t  row_el = (size_t)dim;

    ggml_init_params params = { ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_rows);
    ggml_set_name(cache, "cache_s_l0");           // triggers the fold at init_tensor
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> X(row_el * n_rows);
    for (size_t i = 0; i < X.size(); i++) X[i] = (float)((int)(i % 13) - 6);  // exact in bf16
    ggml_backend_tensor_set(cache, X.data(), 0, X.size() * sizeof(float));

    std::vector<float> Y(X.size(), -999.0f);
    ggml_backend_tensor_get(cache, Y.data(), 0, Y.size() * sizeof(float));
    bool whole_ok = (X == Y);

    // Sub-range: overwrite rows [1,3) only, leave rows 0 and 3 untouched.
    const int64_t head = 1, cells = 2;
    std::vector<float> Z(row_el * cells);
    for (size_t i = 0; i < Z.size(); i++) Z[i] = (float)(100 + (int)(i % 7));
    ggml_backend_tensor_set(cache, Z.data(), head * row_el * sizeof(float), Z.size() * sizeof(float));

    std::vector<float> W(X.size(), -999.0f);
    ggml_backend_tensor_get(cache, W.data(), 0, W.size() * sizeof(float));
    bool range_ok = true;
    for (int64_t r = 0; r < n_rows; r++) {
        for (size_t c = 0; c < row_el; c++) {
            float expect = (r >= head && r < head + cells) ? Z[(r - head) * row_el + c] : X[r * row_el + c];
            if (W[r * row_el + c] != expect) { range_ok = false; }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    printf("[row-fold round-trip] whole: %s   sub-range: %s\n",
        whole_ok ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m",
        range_ok ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m");
    return whole_ok && range_ok;
}

// Correctness gate for the WKV7 metalium path (flat r/w/k/v reader + folded state, both now the
// unconditional default). r/w/k/v are fed to the kernel WITHOUT their [n_embd,T]->[S,H,T] head
// reshape (the handler passes the un-reshaped parent and the reader addresses each head's
// column-tiles itself); the recurrent state is folded on entry. Each config runs the
// reshape->rwkv_wkv7 graph on metalium and compares against CPU. w is filled in (0,1) (a valid
// decay) so the kernel's synthetic-w clamp is a no-op and CPU is a faithful oracle; a,b are
// l2-normed (matches the real model and avoids long-seqlen NaN).
static bool test_wkv7_flat_input(ggml_backend_t metalium, ggml_backend_t cpu) {
    struct cfg { int64_t S, H, Tseq, G; const char * name; };
    const cfg cfgs[] = {
        { 64, 64,  1, 1, "L1G1 (decode)" },
        { 64, 64,  8, 1, "L8G1 (decode)" },
        { 64, 64, 20, 1, "L20G1 (partial chunk)" },
        { 64, 64, 32, 1, "L32G1 (chunked)" },
        { 64, 64, 32, 4, "L32G4 (chunked, multi-seq)" },
        { 64, 64, 48, 2, "L48G2 (cross-32, seq-offset)" },
    };

    auto fill_det = [](ggml_tensor * t, uint32_t seed, float lo, float hi) {
        size_t n = ggml_nelements(t);
        std::vector<float> d(n);
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> u(lo, hi);
        for (auto & x : d) x = u(g);
        ggml_backend_tensor_set(t, d.data(), 0, n * sizeof(float));
    };

    // Build + compute the reshape->rwkv_wkv7 graph on `backend`; return the output.
    auto run = [&](ggml_backend_t backend, int64_t S, int64_t H, int64_t T, int64_t G,
                   std::vector<float> & out) -> bool {
        ggml_init_params p = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
        ggml_context * ctx = ggml_init(p);
        // r/w/k/v as flat [n_embd,T] parents + the head reshape the flat path elides on device.
        ggml_tensor * rp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * wp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * kp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * vp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * ap = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);   // a,b stay [S,H,T]
        ggml_tensor * bp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
        ggml_tensor * sp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H, G);
        ggml_tensor * r = ggml_reshape_3d(ctx, rp, S, H, T);
        ggml_tensor * w = ggml_reshape_3d(ctx, wp, S, H, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, kp, S, H, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, vp, S, H, T);
        ggml_tensor * a = ggml_l2_norm(ctx, ap, 1e-7F);
        ggml_tensor * b = ggml_l2_norm(ctx, bp, 1e-7F);
        ggml_tensor * o = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, sp);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) { ggml_free(ctx); return false; }

        // Identical deterministic inputs for both backends (per-role fixed seeds). w in (0,1).
        fill_det(rp, 1001, -1.0f, 1.0f);
        fill_det(wp, 1002,  0.0f, 0.99f);
        fill_det(kp, 1003, -1.0f, 1.0f);
        fill_det(vp, 1004, -1.0f, 1.0f);
        fill_det(ap, 1005, -1.0f, 1.0f);
        fill_det(bp, 1006, -1.0f, 1.0f);
        fill_det(sp, 1007, -1.0f, 1.0f);

        ggml_backend_graph_compute(backend, gf);
        out.resize(ggml_nelements(o));
        ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    };

    bool all_ok = true;
    printf("[wkv7 metalium vs cpu]\n");
    for (const auto & c : cfgs) {
        const int64_t S = c.S, H = c.H, G = c.G, T = c.Tseq * c.G;
        std::vector<float> o_cpu, o_tt;
        bool ran = run(cpu, S, H, T, G, o_cpu) && run(metalium, S, H, T, G, o_tt);

        // Reject a degenerate (constant/near-zero) reference -- nmse would be meaningless.
        double ref_energy = 0.0;
        for (float x : o_cpu) ref_energy += (double)x * x;

        double e = ran ? nmse(o_cpu.data(), o_tt.data(), o_cpu.size()) : 1e9;
        const double tol = 2e-2;   // bf16 device vs f32 CPU
        bool pass = ran && ref_energy > 1e-6 && e < tol;
        all_ok &= pass;
        printf("  %-30s metalium/cpu nmse=%.2e  %s\n",
            c.name, e, pass ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m");
    }
    return all_ok;
}

// Correctness gate for slice_write_region2_folded: the WKV7 region-2 (final state) scattered
// straight into the row-folded cache must round-trip (unfold on read) back to the CPU WKV7's
// region-2. The folded cache canonical flat index f = h*S*S + i*S + j maps to the WKV7 output
// flat index T*C + f (region-2 rows start at row T, col = i*S+j), so cache_canon[f] == o_cpu[T*C+f].
// This exercises the whole chain: WKV7 -> view_1d(region-2) -> cpy(folded cache) -> my scatter.
static bool test_slice_write_folded(ggml_backend_t metalium, ggml_backend_t cpu) {
    const int64_t S = 64, H = 64, G = 1, C = S * H, dim = S * S * H;
    struct cfg { int64_t L; const char * name; };
    const cfg cfgs[] = { { 1, "L1 (decode)" }, { 8, "L8 (decode)" }, { 32, "L32 (chunked)" } };

    auto fill_det = [](ggml_tensor * t, uint32_t seed, float lo, float hi) {
        size_t n = ggml_nelements(t);
        std::vector<float> d(n);
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> u(lo, hi);
        for (auto & x : d) x = u(g);
        ggml_backend_tensor_set(t, d.data(), 0, n * sizeof(float));
    };

    // CPU: full WKV7 output (region-1 + region-2). Metalium: same WKV7, but region-2 is cpy'd into
    // a folded cache and we read the cache back (canonical). out_cache holds [dim], out_full [C*(T+S*G)].
    auto run = [&](ggml_backend_t backend, int64_t T, bool via_cache,
                   std::vector<float> & out) -> bool {
        ggml_init_params p = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * rp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * wp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * kp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * vp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * H, T);
        ggml_tensor * ap = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
        ggml_tensor * bp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T);
        ggml_tensor * sp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H, G);
        ggml_tensor * r = ggml_reshape_3d(ctx, rp, S, H, T);
        ggml_tensor * w = ggml_reshape_3d(ctx, wp, S, H, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, kp, S, H, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, vp, S, H, T);
        ggml_tensor * a = ggml_l2_norm(ctx, ap, 1e-7F);
        ggml_tensor * b = ggml_l2_norm(ctx, bp, 1e-7F);
        ggml_tensor * o = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, sp);

        ggml_tensor * cache = nullptr;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        if (via_cache) {
            // region-2 = view_1d(o, dim*G) at row offset T (byte offset T*C*4).
            ggml_tensor * region2 = ggml_view_1d(ctx, o, dim * G, (size_t)C * T * sizeof(float));
            cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H, G);
            ggml_set_name(cache, "cache_s_l0");   // folded at init
            ggml_tensor * cv = ggml_view_1d(ctx, cache, dim * G, 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, region2, cv));
        } else {
            ggml_build_forward_expand(gf, o);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) { ggml_free(ctx); return false; }

        fill_det(rp, 1001, -1.0f, 1.0f);
        fill_det(wp, 1002,  0.0f, 0.99f);
        fill_det(kp, 1003, -1.0f, 1.0f);
        fill_det(vp, 1004, -1.0f, 1.0f);
        fill_det(ap, 1005, -1.0f, 1.0f);
        fill_det(bp, 1006, -1.0f, 1.0f);
        fill_det(sp, 1007, -1.0f, 1.0f);

        ggml_backend_graph_compute(backend, gf);
        if (via_cache) {
            out.resize(dim * G);
            ggml_backend_tensor_get(cache, out.data(), 0, out.size() * sizeof(float));
        } else {
            out.resize(ggml_nelements(o));
            ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    };

    bool all_ok = true;
    printf("[slice_write_folded: WKV7 region-2 -> folded cache round-trip]\n");
    for (const auto & c : cfgs) {
        const int64_t T = c.L * G;
        std::vector<float> o_cpu, cache_tt;
        bool ran = run(cpu, T, /*via_cache*/ false, o_cpu) &&
                   run(metalium, T, /*via_cache*/ true, cache_tt);
        // Reference region-2 = o_cpu[T*C .. T*C+dim).
        std::vector<float> ref(dim * G);
        if (ran) for (int64_t f = 0; f < dim * G; f++) ref[f] = o_cpu[(size_t)C * T + f];

        double ref_energy = 0.0;
        for (float x : ref) ref_energy += (double)x * x;
        double e = ran ? nmse(ref.data(), cache_tt.data(), ref.size()) : 1e9;
        const double tol = 3e-2;
        bool pass = ran && ref_energy > 1e-6 && e < tol;
        all_ok &= pass;
        printf("  %-20s cache/cpu-region2 nmse=%.2e  %s\n",
            c.name, e, pass ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m");
    }
    return all_ok;
}

int main(int argc, char ** argv)
{
    (void)argc;
    (void)argv;
    ggml_backend_t cpu = ggml_backend_cpu_init();

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("Metalium");
    if(reg == NULL) {
        fprintf(stderr, "Cannot find the Metalium backend. Is the Meralium backend disabled?\n");
        return 1;
    }
    if(ggml_backend_reg_dev_count(reg) == 0) {
        fprintf(stderr, "No devices found for Metalium backend. Is the kernel driver working?\n");
        return 1;
    }
    ggml_backend_t metalium = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), NULL);

    bool row_fold_ok = test_row_fold_roundtrip(metalium);
    bool wkv7_flat_ok = test_wkv7_flat_input(metalium, cpu);
    bool swf_ok = test_slice_write_folded(metalium, cpu);

    std::vector<std::unique_ptr<test_case>> tests;
    // add_unittests(tests);

    ///////////////// put experiment code here /////////////////
    // easier on the eye to find it (also one line to disable UT)
    // Generic elementwise viewed fusion (compiler.cpp match_elemwise_view -> ttprm::mul/add/sub).
    // UPSTREAM: one operand is a flat->head reshape the route absorbs as a producer-View (this is the
    // RWKV7 kk*a shape -- the relayout we kill). hc>1 so it is a genuine per-head op.
    for (const char * which : { "mul", "add", "sub" }) {
        const std::string op = which;
        tests.push_back(make_test([op](ggml_context* ctx) {
            const int hs = 64, hc = 64, nt = 4;
            ggml_tensor* head = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, hc, nt);
            ggml_tensor* flat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs * hc, nt);
            ggml_tensor* rsh  = ggml_reshape_3d(ctx, flat, hs, hc, nt);
            if (op == "add") return ggml_add(ctx, head, rsh);
            if (op == "sub") return ggml_sub(ctx, head, rsh);
            return ggml_mul(ctx, head, rsh);
        }, "Elementwise viewed " + op + " (upstream reshape)", 2e-2));
    }
    // DOWNSTREAM (head output flattened back to [n_embd, nt]) is NOT routed yet: relabeling the head-grid
    // ttprm result to flat with ttnn::reshape is a real head->flat retile that corrupts for nt>1. It needs
    // ttprm to write straight into the flat tile layout via an `out` View. Left here as the next step:
    // tests.push_back(make_test([](ggml_context* ctx) {
    //     const int hs = 64, hc = 64, nt = 4;
    //     ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, hc, nt);
    //     ggml_tensor* y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, hc, nt);
    //     return ggml_reshape_2d(ctx, ggml_mul(ctx, x, y), hs * hc, nt);
    // }, "Elementwise viewed mul (downstream reshape)", 2e-2));

    // Head-grid norm + per-channel affine (compiler.cpp match_head_affine -> head-grid layer_norm + two
    // group-broadcast ttprm mul/add). This is RWKV7's ln_x: per-head NORM then a per-(head,lane) [n_embd]
    // affine. Requires GGML_METALIUM_TTPRM_AFFINE=1 at runtime (the route is env-gated). nt=1 exercises the
    // ttprm head-grid path with no broadcast (decode); nt=4 hits the token-group broadcast (or falls back).
    for (const int nt : { 1, 4 }) {
        tests.push_back(make_test([nt](ggml_context* ctx) {
            const int hs = 64, hc = 64;
            ggml_tensor* x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hs * hc, nt);
            ggml_tensor* w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hs * hc);
            ggml_tensor* b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hs * hc);
            ggml_tensor* xh   = ggml_reshape_3d(ctx, x, hs, hc, nt);
            ggml_tensor* nrm  = ggml_norm(ctx, xh, 64e-5f);
            ggml_tensor* flat = ggml_reshape_2d(ctx, nrm, hs * hc, nt);
            return ggml_add(ctx, ggml_mul(ctx, flat, w), b);
        }, "Head norm + per-channel affine (nt=" + std::to_string(nt) + ")", 2e-2));
    }
    ///////////////// end of experiment code /////////////////

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
    double not_supported_ratio = (double)not_supported / total_tests;
    std::cout << "\nStats for Metalium backend: " << (total_tests != passed_tests ? "\033[1;31mFAIL\033[0m\n" : "\033[1;32mOK\033[0m\n")
        << "  Test status: " << passed_tests << " / " << total_tests << " passed\n"
        << "  Failed: " << total_tests - passed_tests - not_supported << " (" << std::round(1.0 - passed_ratio - not_supported_ratio) * 100 << "%)\n"
        << "  Not supported: " << not_supported << "\n";

    bool failed = (total_tests != passed_tests) || !row_fold_ok || !wkv7_flat_ok || !swf_ok;
    if(failed) {
        std::cout << "Some tests failed\n";
    }

    ggml_backend_free(metalium);
    ggml_backend_free(cpu);

    return (int)failed;
}
