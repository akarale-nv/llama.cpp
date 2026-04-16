#include "s3gen_ops.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Activation functions
// ============================================================================

ggml_tensor* apply_mish(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* exp_x = ggml_exp(ctx, x);
    ggml_tensor* one_plus_exp = ggml_add(ctx, exp_x, ggml_arange(ctx, 1.0f, 2.0f, 1));
    ggml_tensor* softplus = ggml_log(ctx, one_plus_exp);
    ggml_tensor* tanh_softplus = ggml_tanh(ctx, softplus);
    return ggml_mul(ctx, x, tanh_softplus);
}


ggml_tensor* apply_glu(ggml_context* ctx, ggml_tensor* x, int dim) {
    int64_t half = x->ne[dim] / 2;
    if (dim == 1) {
        ggml_tensor* a = ggml_view_2d(ctx, x, x->ne[0], half, x->nb[1], 0);
        ggml_tensor* b = ggml_view_2d(ctx, x, x->ne[0], half, x->nb[1], half * x->nb[1]);
        return ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
    }
    fprintf(stderr, "GLU: Only dim=1 is currently supported\n");
    return x;
}

ggml_tensor* apply_snake_activation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha_tensor) {
    ggml_tensor* alpha = ggml_reshape_3d(ctx, alpha_tensor, 1, alpha_tensor->ne[0], 1);
    ggml_tensor* alpha_broadcast = ggml_repeat(ctx, alpha, x);
    ggml_tensor* alpha_x = ggml_mul(ctx, alpha_broadcast, x);
    ggml_tensor* sin2_alpha_x = ggml_sqr(ctx, ggml_sin(ctx, alpha_x));
    ggml_tensor* inv_alpha = ggml_div(ctx,
        ggml_repeat(ctx, ggml_arange(ctx, 1.0f, 1.0f + 1, 10.0f), alpha_broadcast),
        alpha_broadcast);
    return ggml_add(ctx, x, ggml_mul(ctx, inv_alpha, sin2_alpha_x));
}

// ============================================================================
// Normalization
// ============================================================================

ggml_tensor* apply_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, weight);
    if (bias) {
        x = ggml_add(ctx, x, bias);
    }
    return x;
}

ggml_tensor* apply_weight_norm(ggml_context* ctx, ggml_tensor* weight_v, ggml_tensor* weight_g) {
    ggml_tensor* weight_v_sqr = ggml_sqr(ctx, weight_v);
    auto weight_f32 = ggml_cast(ctx, weight_v_sqr, GGML_TYPE_F32);
    ggml_tensor* sum_dim0 = ggml_sum_rows(ctx, weight_f32);
    ggml_tensor* reshaped = ggml_reshape_3d(ctx, sum_dim0, sum_dim0->ne[1], sum_dim0->ne[2], 1);
    ggml_tensor* norm_sqr = ggml_sum_rows(ctx, reshaped);
    norm_sqr = ggml_reshape_4d(ctx, norm_sqr, 1, 1, norm_sqr->ne[1], 1);
    ggml_tensor* norm = ggml_sqrt(ctx, norm_sqr);
    ggml_tensor* normalized = ggml_div(ctx, weight_v, norm);
    ggml_tensor* weight_g_reshaped = ggml_reshape_4d(ctx, weight_g, 1, 1, weight_g->ne[0], 1);
    ggml_tensor* weight = ggml_mul(ctx, normalized, weight_g_reshaped);
    if (weight->type != GGML_TYPE_F32) {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    }
    return weight;
}

// ============================================================================
// Convolution helpers
// ============================================================================

ggml_tensor* apply_conv1d(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias,
    int stride, int pad, int dilation, int batch
) {
    if (weight->type != GGML_TYPE_F32) {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    }
    x = ggml_conv_1d(ctx, weight, x, stride, pad, dilation);
    if (batch > 1) {
        x = ggml_reshape_3d(ctx, x, x->ne[0], batch, x->ne[1]);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    }
    ggml_tensor* conv_bias_reshape = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
    x = ggml_add(ctx, x, conv_bias_reshape);
    return x;
}

ggml_tensor* apply_conv1d_nobatch(
    ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias,
    int stride, int padding, int dilation
) {
    if (weight->type != GGML_TYPE_F32) {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    }
    ggml_tensor* conv_out = ggml_conv_1d(ctx, weight, input, stride, padding, dilation);
    ggml_tensor* bias_reshaped = ggml_reshape_4d(ctx, bias, 1, bias->ne[0], 1, 1);
    return ggml_add(ctx, conv_out, bias_reshaped);
}

ggml_tensor* apply_causal_conv1d(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, int kernel_size
) {
    int padding = kernel_size - 1;
    ggml_tensor* pad_shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, padding, x->ne[1], x->ne[2], x->ne[3]);
    ggml_tensor* pad = ggml_repeat(ctx, ggml_arange(ctx, 0, 1, 1), pad_shape);
    ggml_tensor* padded = ggml_concat(ctx, pad, x, 0);
    return apply_conv1d(ctx, padded, weight, bias, 1, 0, 1, static_cast<int>(padded->ne[2]));
}

// ============================================================================
// Linear
// ============================================================================

ggml_tensor* apply_linear(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias) {
    ggml_tensor* weight_2d = ggml_reshape_2d(ctx, weight, weight->ne[0], weight->ne[1]);
    if (weight_2d->type != GGML_TYPE_F32) {
        weight_2d = ggml_cast(ctx, weight_2d, GGML_TYPE_F32);
    }
    ggml_tensor* matmul = ggml_mul_mat(ctx, weight_2d, input);
    return ggml_add(ctx, matmul, bias);
}

// ============================================================================
// Positional encoding
// ============================================================================

ggml_tensor* sinusoidal_pos_emb(ggml_context* ctx, ggml_tensor* x, int dim, float scale) {
    int half_dim = dim / 2;
    float emb_scale = logf(10000.0f) / static_cast<float>(half_dim - 1);
    ggml_tensor* emb = ggml_arange(ctx, 0.0f, static_cast<float>(half_dim), 1.0f);
    emb = ggml_scale(ctx, emb, -emb_scale);
    emb = ggml_exp_inplace(ctx, emb);
    ggml_tensor* repeat_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half_dim, x->ne[0]);
    emb = ggml_repeat(ctx, emb, repeat_shape);
    x = ggml_cont(ctx, ggml_reshape_4d(ctx, x, 1, x->ne[0], x->ne[1], x->ne[2]));
    emb = ggml_mul(ctx, emb, x);
    emb = ggml_scale(ctx, emb, scale);
    return ggml_concat(ctx, ggml_sin(ctx, emb), ggml_cos(ctx, emb), 0);
}

#define CREATE_REVERSE_TENSOR(tensor_name, size, step) \
    ggml_tensor* shape_tensor##tensor_name = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size); \
    ggml_tensor* ones_tensor##tensor_name = ggml_repeat(ctx, ggml_arange(ctx, 1.0f, 2.0f, 1.0f), shape_tensor##tensor_name); \
    ones_tensor##tensor_name = ggml_scale(ctx, ones_tensor##tensor_name, size - 1); \
    ggml_tensor* tensor_name = ggml_arange(ctx, 0.0f, size, 1.0f);  \
    tensor_name = ggml_sub(ctx, ones_tensor##tensor_name, tensor_name);

static ggml_tensor* create_pe(ggml_context* ctx, int d_model, int x_size) {
    double div_term_scale = -(std::log(10000.0) / d_model);
    CREATE_REVERSE_TENSOR(position_pos, x_size, 1);
    position_pos = ggml_reshape_2d(ctx, position_pos, 1, x_size);
    ggml_tensor* position_neg = ggml_arange(ctx, 1.0f, (float)x_size, 1.0f);
    position_neg = ggml_reshape_2d(ctx, position_neg, 1, x_size - 1);
    ggml_tensor* div_term = ggml_arange(ctx, 0.0f, static_cast<float>(d_model), 2.0f);
    div_term = ggml_scale_inplace(ctx, div_term, static_cast<float>(div_term_scale));
    div_term = ggml_exp_inplace(ctx, div_term);
    ggml_tensor* shape_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, div_term->ne[0], x_size);
    div_term = ggml_repeat(ctx, div_term, shape_tensor);
    auto* pos_div_mul = ggml_mul(ctx, div_term, position_pos);
    auto* sin_vals = ggml_sin(ctx, pos_div_mul);
    auto* cos_vals = ggml_cos(ctx, pos_div_mul);
    sin_vals = ggml_reshape_3d(ctx, sin_vals, 1, sin_vals->ne[0], sin_vals->ne[1]);
    cos_vals = ggml_reshape_3d(ctx, cos_vals, 1, cos_vals->ne[0], cos_vals->ne[1]);
    ggml_tensor* pe_positive = ggml_concat(ctx, sin_vals, cos_vals, 0);
    pe_positive = ggml_reshape_2d(ctx, pe_positive, pe_positive->ne[0] * pe_positive->ne[1], pe_positive->ne[2]);
    pe_positive = ggml_reshape_3d(ctx, pe_positive, pe_positive->ne[0], pe_positive->ne[1], 1);
    div_term = ggml_view_2d(ctx, div_term, div_term->ne[0], div_term->ne[1] - 1, div_term->nb[1], 0);
    auto* neg_pos_div_mul = ggml_mul(ctx, div_term, position_neg);
    neg_pos_div_mul = ggml_scale(ctx, neg_pos_div_mul, -1.0f);
    auto* neg_sin_vals = ggml_sin(ctx, neg_pos_div_mul);
    auto* neg_cos_vals = ggml_cos(ctx, neg_pos_div_mul);
    neg_sin_vals = ggml_reshape_3d(ctx, neg_sin_vals, 1, neg_sin_vals->ne[0], neg_sin_vals->ne[1]);
    neg_cos_vals = ggml_reshape_3d(ctx, neg_cos_vals, 1, neg_cos_vals->ne[0], neg_cos_vals->ne[1]);
    ggml_tensor* pe_negative = ggml_concat(ctx, neg_sin_vals, neg_cos_vals, 0);
    pe_negative = ggml_reshape_2d(ctx, pe_negative, pe_negative->ne[0] * pe_negative->ne[1], pe_negative->ne[2]);
    pe_negative = ggml_reshape_3d(ctx, pe_negative, pe_negative->ne[0], pe_negative->ne[1], 1);
    ggml_tensor* pe = ggml_concat(ctx, pe_positive, pe_negative, 1);
    ggml_set_name(pe, "pe");
    return pe;
}

std::pair<ggml_tensor*, ggml_tensor*> espnet_rel_pos_encoding(ggml_context* ctx, ggml_tensor* inputs, int d_model, int max_len) {
    double xscale = std::sqrt(static_cast<double>(d_model));
    int pe_size = std::max(max_len, static_cast<int>(inputs->ne[1]));
    ggml_tensor* pe = create_pe(ctx, d_model, pe_size);
    inputs = ggml_scale(ctx, inputs, static_cast<float>(xscale));
    int64_t slice_lower = (pe->ne[1] / 2) - inputs->ne[1] + 1;
    int64_t slice_upper = (pe->ne[1] / 2) + inputs->ne[1];
    int64_t slice_length = slice_upper - slice_lower;
    auto* pos_emb = ggml_view_3d(ctx, pe, pe->ne[0], slice_length, 1,
        pe->nb[1], pe->nb[2], slice_lower * pe->ne[0] * ggml_element_size(pe));
    ggml_set_name(pos_emb, "pos_emb");
    return { pos_emb, inputs };
}

// ============================================================================
// Misc
// ============================================================================

ggml_tensor* make_pad_mask(ggml_context* ctx, ggml_tensor* lengths, int max_len) {
    int batch_size = static_cast<int>(lengths->ne[0]);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_len, batch_size);
    mask = ggml_repeat(ctx, ggml_arange(ctx, 1, 2, 1), mask);
    return mask;
}
