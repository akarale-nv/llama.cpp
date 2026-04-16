#include "encoder.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>


#ifdef GGML_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

// ============================================================================
// RelPositionMultiHeadedAttention Implementation
// ============================================================================

RelPositionMultiHeadedAttention::RelPositionMultiHeadedAttention(int n_head, int n_feat)
    : d_k(n_feat / n_head), h(n_head) {}

RelPositionMultiHeadedAttention::~RelPositionMultiHeadedAttention() {}

void RelPositionMultiHeadedAttention::load_weights(ggml_context* ctx, const char* prefix) {
    char name_buf[256];
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_q_weight", prefix);
    self_attn_linear_q_weight = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_q_bias", prefix);
    self_attn_linear_q_bias = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_k_weight", prefix);
    self_attn_linear_k_weight = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_k_bias", prefix);
    self_attn_linear_k_bias = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_v_weight", prefix);
    self_attn_linear_v_weight = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_v_bias", prefix);
    self_attn_linear_v_bias = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_out_weight", prefix);
    self_attn_linear_out_weight = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_out_bias", prefix);
    self_attn_linear_out_bias = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_linear_pos_weight", prefix);
    self_attn_linear_pos_weight = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_pos_bias_u", prefix);
    self_attn_pos_bias_u = ggml_get_tensor(ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sself_attn_pos_bias_v", prefix);
    self_attn_pos_bias_v = ggml_get_tensor(ctx, name_buf);
}

ggml_tensor* RelPositionMultiHeadedAttention::rel_shift(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* shape_tensor = ggml_new_tensor_4d(ctx, x->type, 1, x->ne[1], x->ne[2], x->ne[3]);
    ggml_tensor* zero_pad = ggml_repeat(ctx, ggml_arange(ctx, 0, 1, 1), shape_tensor);
    ggml_tensor* x_padded = ggml_concat(ctx, zero_pad, x, 0);
    x_padded = ggml_reshape_4d(ctx, x_padded, x->ne[1], x->ne[0] + 1, x->ne[2], x->ne[3]);
    x_padded = ggml_cont(ctx, ggml_view_4d(ctx, x_padded, x_padded->ne[0], x_padded->ne[1] - 1, x_padded->ne[2], x_padded->ne[3],
        x_padded->nb[1], x_padded->nb[2], x_padded->nb[3], x_padded->nb[1]));
    x_padded = ggml_reshape_4d(ctx, x_padded, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    return ggml_view_4d(ctx, x_padded, (x_padded->ne[0] / 2) + 1, x_padded->ne[1], x_padded->ne[2], x_padded->ne[3],
        x_padded->nb[1], x_padded->nb[2], x_padded->nb[3], 0);
}

ggml_tensor* RelPositionMultiHeadedAttention::forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask, ggml_tensor* pos_emb) {
    ggml_tensor* q = ggml_mul_mat(ctx, self_attn_linear_q_weight, x);
    ggml_tensor* k = ggml_mul_mat(ctx, self_attn_linear_k_weight, x);
    ggml_tensor* v = ggml_mul_mat(ctx, self_attn_linear_v_weight, x);
    int64_t n_batch_pos = pos_emb->ne[2];
    int64_t head_dim = ggml_nelements(q) / (n_batch_pos * h * d_k);
    q = ggml_cont(ctx, ggml_reshape_4d(ctx, ggml_add(ctx, q, self_attn_linear_q_bias), d_k, h, head_dim, n_batch_pos));
    k = ggml_cont(ctx, ggml_reshape_4d(ctx, ggml_add(ctx, k, self_attn_linear_k_bias), d_k, h, head_dim, n_batch_pos));
    v = ggml_cont(ctx, ggml_reshape_4d(ctx, ggml_add(ctx, v, self_attn_linear_v_bias), d_k, h, head_dim, n_batch_pos));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor* p = ggml_mul_mat(ctx, self_attn_linear_pos_weight, pos_emb);
    head_dim = ggml_nelements(p) / (n_batch_pos * h * d_k);
    p = ggml_reshape_4d(ctx, p, d_k, h, head_dim, n_batch_pos);
    p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));
    double attn_scale = 1.0 / sqrt(d_k);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor* matrix_ac = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, q, k)));
    ggml_tensor* matrix_bd = ggml_mul_mat(ctx, p, q);
    if (matrix_ac->ne[0] != matrix_bd->ne[0]) {
        matrix_bd = ggml_cont(ctx, rel_shift(ctx, matrix_bd));
    }
    ggml_tensor* score = ggml_scale(ctx, ggml_add(ctx, matrix_ac, matrix_bd), static_cast<float>(attn_scale));
    if (mask) {
        ggml_tensor* mask_bias = ggml_scale(ctx,
            ggml_sub(ctx, ggml_repeat(ctx, ggml_arange(ctx, 1.0f, 2.0f, 1.0f), mask), mask),
            -10000.0f);
        score = ggml_add(ctx, score, mask_bias);
    }
    score = ggml_soft_max_inplace(ctx, score);
    score = ggml_mul_mat(ctx, v, score);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 0, 2, 1, 3));
    int64_t time1 = ggml_nelements(score) / (score->ne[3] * d_k * h);
    score = ggml_cont(ctx, ggml_reshape_3d(ctx, score, d_k * h, time1, score->ne[3]));
    score = ggml_mul_mat(ctx, self_attn_linear_out_weight, score);
    score = ggml_add(ctx, score, self_attn_linear_out_bias);
    return score;
}

// ============================================================================
// Upsample1D Implementation
// ============================================================================

Upsample1D::Upsample1D(int channels_, int out_channels_, int stride_)
    : channels(channels_), out_channels(out_channels_), stride(stride_),
      conv_weight(nullptr), conv_bias(nullptr) {
}

std::pair<ggml_tensor*, ggml_tensor*> Upsample1D::forward(
    ggml_context* ctx,
    ggml_tensor* inputs,
    ggml_tensor* input_lengths
) {
    // Python equivalent:
    // outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
    // outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
    // outputs = self.conv(outputs)
    
    // Step 1: Interpolate (upsample) - nearest neighbor
    // Input: (batch, channels, time)
    // Output: (batch, channels, time * stride)
    int batch = static_cast<int>(inputs->ne[2]);
    int time = static_cast<int>(inputs->ne[1]);
    int upsampled_time = time * stride;
    
    // Create upsampled tensor using repeat
    ggml_tensor* upsampled = ggml_upscale_ext(ctx, inputs, static_cast<int>(inputs->ne[0]) * stride, static_cast<int>(inputs->ne[1]), static_cast<int>(inputs->ne[2]), static_cast<int>(inputs->ne[3]), GGML_SCALE_MODE_NEAREST);
    ggml_set_name(upsampled, "upsample1d_interpolate");
    
    // Step 2: Pad (left padding of stride * 2)
    int pad_left = stride * 2;
    ggml_tensor* pad_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pad_left, upsampled->ne[1], upsampled->ne[2]);
    ggml_tensor* pad = ggml_repeat(ctx, ggml_arange(ctx, 0, 1, 1), pad_shape);
    ggml_tensor* padded = ggml_concat(ctx, pad_shape, upsampled, 0);
    ggml_set_name(padded, "upsample1d_padded");
    
    // Step 3: Apply convolution
    // Conv1d with kernel_size = stride * 2 + 1, stride = 1, padding = 0
    // In GGML, conv1d operates on (batch, channels, time)
    ggml_tensor* cw = (conv_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv_weight, GGML_TYPE_F32) : conv_weight;
    ggml_tensor* output = ggml_conv_1d(ctx, cw, padded, 1, 0, 1);
    if (conv_bias != nullptr) {
        auto* conv_bias_reshape = ggml_cont(ctx, ggml_transpose(ctx, conv_bias));
        output = ggml_add(ctx, output, conv_bias_reshape);
    }
    ggml_set_name(output, "upsample1d_output");
    
    return {output, nullptr};
}

void Upsample1D::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name_buf[256];
    
    snprintf(name_buf, sizeof(name_buf), "%sconv_weight", prefix);
    conv_weight = ggml_get_tensor(model_ctx, name_buf);
    
    snprintf(name_buf, sizeof(name_buf), "%sconv_bias", prefix);
    conv_bias = ggml_get_tensor(model_ctx, name_buf);
}

// ============================================================================
// PreLookaheadLayer Implementation
// ============================================================================

PreLookaheadLayer::PreLookaheadLayer(int channels_, int pre_lookahead_len_)
    : channels(channels_), pre_lookahead_len(pre_lookahead_len_),
      conv1_weight(nullptr), conv1_bias(nullptr),
      conv2_weight(nullptr), conv2_bias(nullptr) {
}

ggml_tensor* PreLookaheadLayer::forward(ggml_context* ctx, ggml_tensor* inputs, ggml_tensor* mask) {
    // Input: (batch, seq_len, channels)
    // Need to transpose to (batch, channels, seq_len) for conv1d
    ggml_tensor* x = ggml_permute(ctx, inputs, 1, 0, 2, 3);  // (batch, channels, seq_len)
    x = ggml_cont(ctx, x);
    ggml_set_name(x, "pre_lookahead_transpose");
    // Pad right by pre_lookahead_len
    ggml_tensor* x_padded = ggml_pad(ctx, x, pre_lookahead_len, 0, 0, 0);
    ggml_set_name(x_padded, "pre_lookahead_pad1");
    
    // Conv1
    ggml_tensor* c1w = (conv1_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv1_weight, GGML_TYPE_F32) : conv1_weight;
    ggml_tensor* x_conv1 = ggml_conv_1d(ctx, c1w, x_padded, 1, 0, 1);
    if (conv1_bias != nullptr) {
        auto* conv1_bias_reshape = ggml_reshape_2d(ctx, conv1_bias, 1, conv1_bias->ne[0]);
        x_conv1 = ggml_add(ctx, x_conv1, conv1_bias_reshape);
    }
    ggml_set_name(x_conv1, "pre_lookahead_conv1");

    // LeakyReLU
    ggml_tensor* x_relu = apply_leaky_relu(ctx, x_conv1, 0.01f);
    ggml_set_name(x_relu, "pre_lookahead_relu");
    
    // Pad left by 2
    ggml_tensor* left_pad = ggml_new_tensor_4d(ctx, x_relu->type, 2, x_relu->ne[1], x_relu->ne[2], x_relu->ne[3]);
    left_pad = ggml_repeat(ctx, ggml_arange(ctx, 0, 1, 1), left_pad);

    ggml_tensor* x_padded2 = ggml_concat(ctx, left_pad, x_relu, 0);
    ggml_set_name(x_padded2, "pre_lookahead_pad2");

    // Conv2
    ggml_tensor* c2w = (conv2_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv2_weight, GGML_TYPE_F32) : conv2_weight;
    ggml_tensor* x_conv2 = ggml_conv_1d(ctx, c2w, x_padded2, 1, 0, 1);
    if (conv2_bias != nullptr) {
        auto* conv2_bias_reshape = ggml_reshape_2d(ctx, conv2_bias, 1, conv2_bias->ne[0]);
        x_conv2 = ggml_add(ctx, x_conv2, conv2_bias_reshape);
    }
    ggml_set_name(x_conv2, "pre_lookahead_conv2");
    
    // Transpose back to (batch, seq_len, channels)
    ggml_tensor* output = ggml_permute(ctx, x_conv2, 1, 0, 2, 3);
    output = ggml_cont(ctx, output);
    ggml_set_name(output, "pre_lookahead_transpose_back");
    
    // Residual connection
    ggml_tensor* result = ggml_add(ctx, output, inputs);
    ggml_set_name(result, "pre_lookahead_residual");
    
    // Apply mask to zero out padding positions
    if (mask) {
        ggml_tensor* mask_3d = ggml_reshape_3d(ctx, mask, 1, mask->ne[0], 1);
        result = ggml_mul(ctx, result, mask_3d);
    }
    
    return result;
}

void PreLookaheadLayer::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name_buf[256];
    
    snprintf(name_buf, sizeof(name_buf), "%sconv1_weight", prefix);
    conv1_weight = ggml_get_tensor(model_ctx, name_buf);
    
    snprintf(name_buf, sizeof(name_buf), "%sconv1_bias", prefix);
    conv1_bias = ggml_get_tensor(model_ctx, name_buf);
    
    snprintf(name_buf, sizeof(name_buf), "%sconv2_weight", prefix);
    conv2_weight = ggml_get_tensor(model_ctx, name_buf);
    
    snprintf(name_buf, sizeof(name_buf), "%sconv2_bias", prefix);
    conv2_bias = ggml_get_tensor(model_ctx, name_buf);
}

// ============================================================================
// ConformerEncoderLayer Implementation
// ============================================================================

ConformerEncoderLayer::ConformerEncoderLayer(
    int attention_heads_, int size_, bool normalize_before_, 
    bool has_macaron_, bool has_conv_module_)
    : size(size_), normalize_before(normalize_before_), 
      has_macaron(has_macaron_), has_conv_module(has_conv_module_),
      ff_scale(has_macaron_ ? 0.5f : 1.0f) {
    
    // Initialize all pointers to nullptr
    norm_ff_weight = norm_ff_bias = nullptr;
    norm_mha_weight = norm_mha_bias = nullptr;
    norm_ff_macaron_weight = norm_ff_macaron_bias = nullptr;
    norm_conv_weight = norm_conv_bias = nullptr;
    norm_final_weight = norm_final_bias = nullptr;
    
    feed_forward_w_1_weight = feed_forward_w_1_bias = nullptr;
    feed_forward_w_2_weight = feed_forward_w_2_bias = nullptr;
    
    feed_forward_macaron_w_1_weight = feed_forward_macaron_w_1_bias = nullptr;
    feed_forward_macaron_w_2_weight = feed_forward_macaron_w_2_bias = nullptr;
    
    conv_pointwise_conv1_weight = conv_pointwise_conv1_bias = nullptr;
    conv_depthwise_conv_weight = conv_depthwise_conv_bias = nullptr;
    conv_norm_weight = conv_norm_bias = nullptr;
    conv_pointwise_conv2_weight = conv_pointwise_conv2_bias = nullptr;

    self_attn = new RelPositionMultiHeadedAttention(attention_heads_, size_);

}

ggml_tensor* ConformerEncoderLayer::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* mask,
    ggml_tensor* pos_emb,
    ggml_tensor* mask_pad
) {
    // Python equivalent from ConformerEncoderLayer.forward()
    
    // 2. Multi-headed self-attention
    {
        ggml_tensor* residual = x;
        
        if (normalize_before) 
        {
            x = ggml_norm(ctx, x, 1e-12f);
            x = ggml_mul(ctx, x, norm_mha_weight);
            x = ggml_add(ctx, x, norm_mha_bias);
        }
        ggml_tensor* attn_output = self_attn->forward(ctx, x, mask, pos_emb);
        // Residual
        x = ggml_add(ctx, residual, attn_output);
        
        if (!normalize_before) {
            x = ggml_norm(ctx, x, 1e-12f);
            x = ggml_mul(ctx, x, norm_mha_weight);
            x = ggml_add(ctx, x, norm_mha_bias);
        }
    }
    
    // 3. Optional: Convolution module
    if (has_conv_module && conv_pointwise_conv1_weight != nullptr) {
        ggml_tensor* residual = x;
        
        if (normalize_before) {
            x = ggml_norm(ctx, x, 1e-12f);
            x = ggml_mul(ctx, x, norm_conv_weight);
            x = ggml_add(ctx, x, norm_conv_bias);
        }
        
        // Transpose to (batch, channels, time)
        ggml_tensor* x_t = ggml_permute(ctx, x, 0, 2, 1, 3);
        
        // Pointwise conv 1
        ggml_tensor* pw1 = (conv_pointwise_conv1_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv_pointwise_conv1_weight, GGML_TYPE_F32) : conv_pointwise_conv1_weight;
        ggml_tensor* conv_out = ggml_conv_1d(ctx, pw1, x_t, 1, 0, 1);
        conv_out = ggml_add(ctx, conv_out, conv_pointwise_conv1_bias);

        // GLU activation
        conv_out = apply_glu(ctx, conv_out, 1);

        // Depthwise conv
        ggml_tensor* dw = (conv_depthwise_conv_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv_depthwise_conv_weight, GGML_TYPE_F32) : conv_depthwise_conv_weight;
        conv_out = ggml_conv_1d(ctx, dw, conv_out, 1, 0, 1);
        conv_out = ggml_add(ctx, conv_out, conv_depthwise_conv_bias);

        // Normalization (batch norm or layer norm)
        conv_out = ggml_norm(ctx, conv_out, 1e-5f);
        conv_out = ggml_mul(ctx, conv_out, conv_norm_weight);
        conv_out = ggml_add(ctx, conv_out, conv_norm_bias);

        // Activation
        conv_out = apply_swish(ctx, conv_out);

        // Pointwise conv 2
        ggml_tensor* pw2 = (conv_pointwise_conv2_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv_pointwise_conv2_weight, GGML_TYPE_F32) : conv_pointwise_conv2_weight;
        conv_out = ggml_conv_1d(ctx, pw2, conv_out, 1, 0, 1);
        conv_out = ggml_add(ctx, conv_out, conv_pointwise_conv2_bias);
        
        // Transpose back
        conv_out = ggml_permute(ctx, conv_out, 0, 2, 1, 3);
        
        // Residual
        x = ggml_add(ctx, residual, conv_out);
        
        if (!normalize_before) {
            x = ggml_norm(ctx, x, 1e-12f);
            x = ggml_mul(ctx, x, norm_conv_weight);
            x = ggml_add(ctx, x, norm_conv_bias);
        }
    }
    
    // 4. Feed-forward module
    {
        ggml_tensor* residual = x;
        
        if (normalize_before) {
            x = ggml_norm(ctx, x, 1e-12f);
            x = ggml_mul(ctx, x, norm_ff_weight);
            x = ggml_add(ctx, x, norm_ff_bias);
        }
        // Feed-forward
        ggml_tensor* ff_out = ggml_mul_mat(ctx, feed_forward_w_1_weight, x);
        ff_out = ggml_add(ctx, ff_out, feed_forward_w_1_bias);
        ff_out = ggml_silu(ctx, ff_out);
        ff_out = ggml_mul_mat(ctx, feed_forward_w_2_weight, ff_out);
        ff_out = ggml_add(ctx, ff_out, feed_forward_w_2_bias);
        
        // Residual with scaling
        x = ggml_add(ctx, residual, ggml_scale(ctx, ff_out, ff_scale));

        if (!normalize_before) {
            x = ggml_norm(ctx, x, 1e-12f);
            x = ggml_mul(ctx, x, norm_ff_weight);
            x = ggml_add(ctx, x, norm_ff_bias);
        }
    }
    
    // 5. Optional: Final normalization (for conv module)
    if (has_conv_module && norm_final_weight != nullptr) {
        x = ggml_norm(ctx, x, 1e-12f);
        x = ggml_mul(ctx, x, norm_final_weight);
        x = ggml_add(ctx, x, norm_final_bias);
    }
    
    // Apply mask_pad to zero out padding positions
    // mask_pad shape: (T, 1) or broadcastable, x shape: (T, hidden_dim, batch)
    if (mask_pad) {
        ggml_tensor* mask_3d = ggml_reshape_3d(ctx, mask_pad, 1, mask_pad->ne[0], 1);
        x = ggml_mul(ctx, x, mask_3d);
    }
    
    return x;
}

void ConformerEncoderLayer::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name_buf[512];
    
    // Layer norms
    snprintf(name_buf, sizeof(name_buf), "%snorm_ff_weight", prefix);
    norm_ff_weight = ggml_get_tensor(model_ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%snorm_ff_bias", prefix);
    norm_ff_bias = ggml_get_tensor(model_ctx, name_buf);
    
    snprintf(name_buf, sizeof(name_buf), "%snorm_mha_weight", prefix);
    norm_mha_weight = ggml_get_tensor(model_ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%snorm_mha_bias", prefix);
    norm_mha_bias = ggml_get_tensor(model_ctx, name_buf);
    
    if (has_macaron) {
        snprintf(name_buf, sizeof(name_buf), "%snorm_ff_macaron_weight", prefix);
        norm_ff_macaron_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%snorm_ff_macaron_bias", prefix);
        norm_ff_macaron_bias = ggml_get_tensor(model_ctx, name_buf);
    }
    
    if (has_conv_module) {
        snprintf(name_buf, sizeof(name_buf), "%snorm_conv_weight", prefix);
        norm_conv_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%snorm_conv_bias", prefix);
        norm_conv_bias = ggml_get_tensor(model_ctx, name_buf);
        
        snprintf(name_buf, sizeof(name_buf), "%snorm_final_weight", prefix);
        norm_final_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%snorm_final_bias", prefix);
        norm_final_bias = ggml_get_tensor(model_ctx, name_buf);
    }
    
    // Self-attention
    self_attn->load_weights(model_ctx, prefix);
    
    // Feed-forward
    snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_w_1_weight", prefix);
    feed_forward_w_1_weight = ggml_get_tensor(model_ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_w_1_bias", prefix);
    feed_forward_w_1_bias = ggml_get_tensor(model_ctx, name_buf);
    
    snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_w_2_weight", prefix);
    feed_forward_w_2_weight = ggml_get_tensor(model_ctx, name_buf);
    snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_w_2_bias", prefix);
    feed_forward_w_2_bias = ggml_get_tensor(model_ctx, name_buf);
    
    // Macaron feed-forward (if present)
    if (has_macaron) {
        snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_macaron_w_1_weight", prefix);
        feed_forward_macaron_w_1_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_macaron_w_1_bias", prefix);
        feed_forward_macaron_w_1_bias = ggml_get_tensor(model_ctx, name_buf);
        
        snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_macaron_w_2_weight", prefix);
        feed_forward_macaron_w_2_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%sfeed_forward_macaron_w_2_bias", prefix);
        feed_forward_macaron_w_2_bias = ggml_get_tensor(model_ctx, name_buf);
    }
    
    // Convolution module (if present)
    if (has_conv_module) {
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_pointwise_conv1_weight", prefix);
        conv_pointwise_conv1_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_pointwise_conv1_bias", prefix);
        conv_pointwise_conv1_bias = ggml_get_tensor(model_ctx, name_buf);
        
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_depthwise_conv_weight", prefix);
        conv_depthwise_conv_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_depthwise_conv_bias", prefix);
        conv_depthwise_conv_bias = ggml_get_tensor(model_ctx, name_buf);
        
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_norm_weight", prefix);
        conv_norm_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_norm_bias", prefix);
        conv_norm_bias = ggml_get_tensor(model_ctx, name_buf);
        
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_pointwise_conv2_weight", prefix);
        conv_pointwise_conv2_weight = ggml_get_tensor(model_ctx, name_buf);
        snprintf(name_buf, sizeof(name_buf), "%sconv_module_pointwise_conv2_bias", prefix);
        conv_pointwise_conv2_bias = ggml_get_tensor(model_ctx, name_buf);
    }
}

// ============================================================================
// UpsampleConformerEncoder Implementation
// ============================================================================

UpsampleConformerEncoder::UpsampleConformerEncoder(const S3GenConfig& cfg)
    : config(cfg), embed_linear_weight(nullptr), embed_linear_bias(nullptr),
      embed_norm_weight(nullptr), embed_norm_bias(nullptr),
      pos_enc_pe(nullptr),
      pre_lookahead_layer(nullptr), up_layer(nullptr),
      up_embed_linear_weight(nullptr), up_embed_linear_bias(nullptr),
      up_embed_norm_weight(nullptr), up_embed_norm_bias(nullptr),
      after_norm_weight(nullptr), after_norm_bias(nullptr) {

    // Initialize layers
    pre_lookahead_layer = new PreLookaheadLayer(config.encoder_output_size, config.pre_lookahead_len);
    up_layer = new Upsample1D(config.encoder_output_size, config.encoder_output_size, config.encoder_upsample_stride);
    embed_d_model = cfg.encoder_output_size;
    embed_max_len = 5000;
    // Initialize encoder layers
    for (int i = 0; i < config.encoder_num_blocks; i++) {
        encoders.push_back(new ConformerEncoderLayer(
            config.encoder_attention_heads,
            config.encoder_output_size,
            config.encoder_normalize_before,
            config.encoder_macaron_style,
            config.encoder_use_cnn_module
        ));
    }

    // Initialize upsampled encoder layers
    for (int i = 0; i < config.encoder_num_up_blocks; i++) {
        up_encoders.push_back(new ConformerEncoderLayer(
            config.encoder_attention_heads,
            config.encoder_output_size,
            config.encoder_normalize_before,
            config.encoder_macaron_style,
            config.encoder_use_cnn_module
        ));
    }
}

UpsampleConformerEncoder::~UpsampleConformerEncoder() {
    // Clean up layer objects
    delete pre_lookahead_layer;
    delete up_layer;
    
    for (auto layer : encoders) {
        delete layer;
    }
    for (auto layer : up_encoders) {
        delete layer;
    }
}

bool UpsampleConformerEncoder::load_model(ggml_context* ggml_ctx)
{
    try {
        printf("UpsampleConformerEncoder: Loading model tensors:\n");

        // Load embedding layer weights
        embed_linear_weight = checked_get_tensor(ggml_ctx, "fenc_embed_out_0_weight");
        embed_linear_bias = checked_get_tensor(ggml_ctx, "fenc_embed_out_0_bias");
        embed_norm_weight = checked_get_tensor(ggml_ctx, "fenc_embed_out_1_weight");
        embed_norm_bias = checked_get_tensor(ggml_ctx, "fenc_embed_out_1_bias");

        // Load pre-lookahead layer
        pre_lookahead_layer->load_weights(ggml_ctx, "fenc_prelook_");

        // Load encoder layers
        for (size_t i = 0; i < encoders.size(); i++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "fenc_blk_%zu_", i);
            encoders[i]->load_weights(ggml_ctx, prefix);
        }

        // Load upsample layer
        up_layer->load_weights(ggml_ctx, "fenc_up_layer_");

        // Load upsampled embedding layer weights
        up_embed_linear_weight = checked_get_tensor(ggml_ctx, "fenc_up_embed_out_0_weight");
        up_embed_linear_bias = checked_get_tensor(ggml_ctx, "fenc_up_embed_out_0_bias");
        up_embed_norm_weight = checked_get_tensor(ggml_ctx, "fenc_up_embed_out_1_weight");
        up_embed_norm_bias = checked_get_tensor(ggml_ctx, "fenc_up_embed_out_1_bias");

        // Load upsampled encoder layers
        for (size_t i = 0; i < up_encoders.size(); i++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "fenc_upblk_%zu_", i);
            up_encoders[i]->load_weights(ggml_ctx, prefix);
        }

        // Load final normalization
        after_norm_weight = checked_get_tensor(ggml_ctx, "fenc_after_norm_weight");
        after_norm_bias = checked_get_tensor(ggml_ctx, "fenc_after_norm_bias");

        printf("UpsampleConformerEncoder: All tensors loaded successfully!\n");
    }
    catch (const std::exception& e) {
        fprintf(stderr, "UpsampleConformerEncoder: %s\n", e.what());
        return false;
    }
    return true;
}


ggml_tensor* UpsampleConformerEncoder::build_graph(
    ggml_context* ctx,
    ggml_tensor* xs,
    ggml_tensor* xs_lens,
    int decoding_chunk_size,
    int num_decoding_left_chunks,
    ggml_tensor* attention_mask,
    ggml_tensor** upsampled_mask_out
) {

    int T = static_cast<int>(xs->ne[1]);  // Time dimension
    
    // 1. Create padding mask - use provided attention_mask if available
    ggml_tensor* masks = nullptr;
    if (attention_mask) {
        // attention_mask shape: (1, total_len) or (total_len, 1)
        // Reshape to (T, 1) for internal use
        masks = ggml_reshape_2d(ctx, attention_mask, T, 1);
    } else {
        // Fall back to computing mask from lengths
        masks = make_pad_mask(ctx, xs_lens, T);
    }

    ggml_tensor* ones_tensor = ggml_repeat(ctx, ggml_arange(ctx, 1.0f, 2.0f, 1.0f), masks);
    ggml_tensor* inv_mask = ggml_add(ctx, masks, ones_tensor);

    // Unsqueeze at dimension 1
    ggml_tensor* masks_3d = ggml_reshape_3d(ctx, masks, 
        1,      // dim 1 (new dimension)
        masks->ne[0],   // dim 0 (stays same)
        masks->ne[1]   // dim 2 (was dim 1)
    );
    ggml_set_name(masks_3d, "masks");
    // 2. Apply embedding layer
    ggml_tensor* x = apply_embed_layer(ctx, xs, 
                                       embed_linear_weight, embed_linear_bias,
                                       embed_norm_weight, embed_norm_bias,
                                       masks_3d);
    print_shape("embedded", x);
    // 3. Compute positional embeddings
    //ggml_tensor* pos_emb = compute_pos_emb(ctx, x);
    auto [pos_emb, y] = espnet_rel_pos_encoding(ctx, x, embed_d_model, embed_max_len);
    ggml_set_name(pos_emb, "pos_emb");
    print_shape("pos_emb", pos_emb);
    // 6. Apply pre-lookahead layer (with mask to zero out padding)
    
    y = pre_lookahead_layer->forward(ctx, y, masks);
    print_shape("after_pre_lookahead", y);
    
    // 7. Forward through main encoder layers
    y = forward_layers(ctx, y, masks, pos_emb, masks);
    print_shape("after_encoders", y);

    // 8. Upsample
    // Need to transpose to (batch, channels, time)
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
    auto [upsampled_y, upsampled_lens] = up_layer->forward(ctx, y, xs_lens);
    y = ggml_cont(ctx, ggml_permute(ctx, upsampled_y, 1, 0, 2, 3));  // Back to (batch, time, channels)
    ggml_set_name(y, "upsampled");
    print_shape("upsampled", y);
    
    // 9. Create new masks for upsampled sequence
    auto T_upsampled = static_cast<int>(y->ne[1]);  // Time dimension after 2x upsample
    
    // Upsample the attention mask by 2x using nearest-neighbor interpolation
    // This effectively does repeat_interleave: [a, b, c] -> [a, a, b, b, c, c]
    ggml_tensor* up_masks = nullptr;
    if (attention_mask) {
        // Use ggml_upscale_ext with nearest-neighbor mode for proper repeat_interleave
        // masks shape: (T, 1) -> up_masks shape: (T*2, 1)
        up_masks = ggml_upscale_ext(ctx, masks, T_upsampled, 1, 1, 1, GGML_SCALE_MODE_NEAREST);
        ggml_set_name(up_masks, "upsampled_attention_mask");
        
        // Output the upsampled mask for decoder use
        if (upsampled_mask_out) {
            *upsampled_mask_out = up_masks;
        }
    } else {
        // Fall back to computing mask from lengths
        up_masks = make_pad_mask(ctx, xs_lens, T_upsampled);
        ones_tensor = ggml_repeat(ctx, ggml_arange(ctx, 1.0f, 2.0f, 1.0f), up_masks);
        up_masks = ggml_add(ctx, up_masks, ones_tensor);
    }

    // Unsqueeze at dimension 1
    ggml_tensor* up_masks_3d = ggml_reshape_3d(ctx, up_masks,
        1,      // dim 1 (new dimension)
        up_masks->ne[0],   // dim 0 (stays same)
        up_masks->ne[1]   // dim 2 (was dim 1)
    );
    ggml_set_name(up_masks_3d, "masks");
    
    // 10. Apply upsampled embedding layer
    y = apply_embed_layer(ctx, y,
                         up_embed_linear_weight, up_embed_linear_bias,
                         up_embed_norm_weight, up_embed_norm_bias,
                         up_masks_3d);
    ggml_set_name(y, "up_embedded");
    print_shape("up_embedded", y);

    auto [pos_emb_upsampled, y_upsampled] = espnet_rel_pos_encoding(ctx, y, embed_d_model, embed_max_len);

    // 14. Forward through upsampled encoder layers
    y_upsampled = forward_up_layers(ctx, y_upsampled, up_masks, pos_emb_upsampled, up_masks);
    ggml_set_name(y_upsampled, "after_up_encoders");
    print_shape("after_up_encoders", y_upsampled);
    ggml_set_name(y_upsampled, "y");
    ggml_set_name(pos_emb_upsampled, "pos_embedded");
    
    y_upsampled = apply_layer_norm(ctx, y_upsampled, after_norm_weight, after_norm_bias, 1e-5f);

    return y_upsampled;
}



void UpsampleConformerEncoder::print_model_info() const {
    printf("UpsampleConformerEncoder Configuration:\n");
    printf("  Input size: %d\n", config.input_size);
    printf("  Output size: %d\n", config.encoder_output_size);
    printf("  Attention heads: %d\n", config.encoder_attention_heads);
    printf("  Linear units: %d\n", config.encoder_linear_units);
    printf("  Number of blocks: %d\n", config.encoder_num_blocks);
    printf("  Number of upsampled blocks: %d\n", config.encoder_num_up_blocks);
    printf("  Upsample stride: %d\n", config.encoder_upsample_stride);
    printf("  Pre-lookahead length: %d\n", config.pre_lookahead_len);
    printf("  Normalize before: %s\n", config.encoder_normalize_before ? "true" : "false");
    printf("  Macaron style: %s\n", config.encoder_macaron_style ? "true" : "false");
    printf("  Use CNN module: %s\n", config.encoder_use_cnn_module ? "true" : "false");
}

ggml_tensor* UpsampleConformerEncoder::checked_get_tensor(
    ggml_context* ctx, const char* name
) {
    ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor) {
        char error_msg[512];
        snprintf(error_msg, sizeof(error_msg), 
                "Tensor '%s' not found in model", name);
        throw std::runtime_error(error_msg);
    }
    return tensor;
}

void UpsampleConformerEncoder::checked_validate_tensor(
    const ggml_tensor* tensor, const char* name
) {
    if (!tensor) {
        char error_msg[512];
        snprintf(error_msg, sizeof(error_msg), 
                "Tensor '%s' is null", name);
        throw std::runtime_error(error_msg);
    }
}

void UpsampleConformerEncoder::print_shape(const char* name, const ggml_tensor* t) const {
    if (!t) {
        printf("  %s: NULL\n", name);
        return;
    }
    printf("  %s: [", name);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)t->ne[i]);
    }
    printf("]\n");
}

