#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <utility>
#include <cstdio>

#define CHECK_TENSOR_LOAD(tensor, msg) if (!tensor) { fprintf(stderr, "Failed to load %s\n", msg); return false; }

// ============================================================================
// Activation functions
// ============================================================================

ggml_tensor* apply_mish(ggml_context* ctx, ggml_tensor* x);
// SiLU(x) = x * sigmoid(x)
inline ggml_tensor* apply_silu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_silu(ctx, x);
}

// Swish(x) = x * sigmoid(x) — identical to SiLU
inline ggml_tensor* apply_swish(ggml_context* ctx, ggml_tensor* x) {
    return ggml_mul(ctx, x, ggml_sigmoid(ctx, x));
}

// ELU(x) = x if x > 0, else exp(x) - 1
inline ggml_tensor* apply_elu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_elu(ctx, x);
}

// LeakyReLU(x) = x if x > 0, else alpha * x
inline ggml_tensor* apply_leaky_relu(ggml_context* ctx, ggml_tensor* x, float alpha = 0.01f) {
    return ggml_leaky_relu(ctx, x, alpha, true);
}
ggml_tensor* apply_glu(ggml_context* ctx, ggml_tensor* x, int dim);
ggml_tensor* apply_snake_activation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha_tensor);

// ============================================================================
// Normalization
// ============================================================================

// Group normalization (eps = 1e-5)
inline ggml_tensor* apply_group_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, int groups) {
    return ggml_group_norm(ctx, x, groups, 1e-5f);
}
ggml_tensor* apply_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, float eps);
ggml_tensor* apply_weight_norm(ggml_context* ctx, ggml_tensor* weight_v, ggml_tensor* weight_g);

// ============================================================================
// Convolution helpers
// ============================================================================

// General conv1d with batch support (used by decoder/encoder)
ggml_tensor* apply_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight,
    ggml_tensor* bias, int stride, int pad, int dilation, int batch);

// Simple conv1d without batch (used by hift vocoder)
ggml_tensor* apply_conv1d_nobatch(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
    ggml_tensor* bias, int stride = 1, int padding = 1, int dilation = 1);

ggml_tensor* apply_causal_conv1d(ggml_context* ctx, ggml_tensor* x,
    ggml_tensor* weight, ggml_tensor* bias, int kernel_size);

// ============================================================================
// Linear
// ============================================================================

ggml_tensor* apply_linear(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias);

// ============================================================================
// Positional encoding
// ============================================================================

ggml_tensor* sinusoidal_pos_emb(ggml_context* ctx, ggml_tensor* x, int dim, float scale = 1000.0f);

std::pair<ggml_tensor*, ggml_tensor*> espnet_rel_pos_encoding(
    ggml_context* ctx, ggml_tensor* inputs, int d_model, int max_len = 5000);

// ============================================================================
// Misc
// ============================================================================

ggml_tensor* make_pad_mask(ggml_context* ctx, ggml_tensor* lengths, int max_len);
