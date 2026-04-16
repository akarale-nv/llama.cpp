#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <vector>
#include <cstdint>
#include "s3gen_config.h"
#include "s3gen_ops.h"

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;


/**
 * Upsample1D Layer
 * A 1D upsampling layer with convolution
 * 
 * Python equivalent:
 *   class Upsample1D(nn.Module):
 *       def __init__(self, channels: int, out_channels: int, stride: int = 2)
 */
struct Upsample1D {
    int channels;
    int out_channels;
    int stride;
    
    // Convolution weights (kernel_size = stride * 2 + 1, stride=1, padding=0)
    ggml_tensor* conv_weight;  // Shape: (out_channels, channels, kernel_size)
    ggml_tensor* conv_bias;    // Shape: (out_channels)
    
    Upsample1D(int channels_, int out_channels_, int stride_);
    
    /**
     * Forward pass
     * @param ctx GGML context for creating new tensors
     * @param inputs Input tensor (batch, channels, time)
     * @param input_lengths Input lengths tensor
     * @return Tuple of (output tensor, output lengths)
     */
    std::pair<ggml_tensor*, ggml_tensor*> forward(
        ggml_context* ctx,
        ggml_tensor* inputs,
        ggml_tensor* input_lengths
    );
    
    // Load weights from model
    void load_weights(ggml_context* model_ctx, const char* prefix);
};

/**
 * PreLookaheadLayer
 * Applies lookahead convolution for autoregressive modeling
 * 
 * Python equivalent:
 *   class PreLookaheadLayer(nn.Module):
 *       def __init__(self, channels: int, pre_lookahead_len: int = 1)
 */
struct PreLookaheadLayer {
    int channels;
    int pre_lookahead_len;
    
    // Conv layers
    ggml_tensor* conv1_weight;  // (channels, channels, pre_lookahead_len + 1)
    ggml_tensor* conv1_bias;    // (channels)
    ggml_tensor* conv2_weight;  // (channels, channels, 3)
    ggml_tensor* conv2_bias;    // (channels)
    
    PreLookaheadLayer(int channels_, int pre_lookahead_len_);
    
    /**
     * Forward pass
     * @param ctx GGML context
     * @param inputs Input tensor (batch, seq_len, channels)
     * @param mask Optional attention mask (1.0 for valid, 0.0 for padding)
     * @return Output tensor with residual connection
     */
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* inputs, ggml_tensor* mask = nullptr);
    
    // Load weights from model
    void load_weights(ggml_context* model_ctx, const char* prefix);
};

struct RelPositionMultiHeadedAttention {
    RelPositionMultiHeadedAttention(int n_head, int n_feat);
    ~RelPositionMultiHeadedAttention();
    void load_weights(ggml_context* ctx, const char* prefix);
    ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* x);
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask, ggml_tensor* pos_emb);
private:
    int d_k;
    int h;
    ggml_tensor* self_attn_linear_q_weight;
    ggml_tensor* self_attn_linear_q_bias;
    ggml_tensor* self_attn_linear_k_weight;
    ggml_tensor* self_attn_linear_k_bias;
    ggml_tensor* self_attn_linear_v_weight;
    ggml_tensor* self_attn_linear_v_bias;
    ggml_tensor* self_attn_linear_out_weight;
    ggml_tensor* self_attn_linear_out_bias;
    ggml_tensor* self_attn_linear_pos_weight;
    ggml_tensor* self_attn_pos_bias_u;
    ggml_tensor* self_attn_pos_bias_v;
};

struct ConformerEncoderLayer {
    int size;
    bool normalize_before;
    bool has_macaron;
    bool has_conv_module;
    float ff_scale;
    RelPositionMultiHeadedAttention* self_attn;
    
    // Layer normalization weights
    ggml_tensor* norm_ff_weight;
    ggml_tensor* norm_ff_bias;
    ggml_tensor* norm_mha_weight;
    ggml_tensor* norm_mha_bias;
    ggml_tensor* norm_ff_macaron_weight;
    ggml_tensor* norm_ff_macaron_bias;
    ggml_tensor* norm_conv_weight;
    ggml_tensor* norm_conv_bias;
    ggml_tensor* norm_final_weight;
    ggml_tensor* norm_final_bias;
    
    // Feed-forward weights
    ggml_tensor* feed_forward_w_1_weight;
    ggml_tensor* feed_forward_w_1_bias;
    ggml_tensor* feed_forward_w_2_weight;
    ggml_tensor* feed_forward_w_2_bias;
    
    // Macaron feed-forward weights (optional)
    ggml_tensor* feed_forward_macaron_w_1_weight;
    ggml_tensor* feed_forward_macaron_w_1_bias;
    ggml_tensor* feed_forward_macaron_w_2_weight;
    ggml_tensor* feed_forward_macaron_w_2_bias;
    
    // Convolution module weights (optional)
    ggml_tensor* conv_pointwise_conv1_weight;
    ggml_tensor* conv_pointwise_conv1_bias;
    ggml_tensor* conv_depthwise_conv_weight;
    ggml_tensor* conv_depthwise_conv_bias;
    ggml_tensor* conv_norm_weight;
    ggml_tensor* conv_norm_bias;
    ggml_tensor* conv_pointwise_conv2_weight;
    ggml_tensor* conv_pointwise_conv2_bias;
    
    ConformerEncoderLayer(int attention_heads_, int size_, bool normalize_before_,
        bool has_macaron_, bool has_conv_module_);
    
    /**
     * Forward pass through the conformer encoder layer
     * @param ctx GGML context
     * @param x Input tensor (batch, time, size)
     * @param mask Attention mask tensor
     * @param pos_emb Positional embedding tensor
     * @param mask_pad Padding mask tensor
     * @return Output tensor
     */
    ggml_tensor* forward(
        ggml_context* ctx,
        ggml_tensor* x,
        ggml_tensor* mask,
        ggml_tensor* pos_emb,
        ggml_tensor* mask_pad
    );
    
    // Load weights from model
    void load_weights(ggml_context* model_ctx, const char* prefix);
};

/**
 * UpsampleConformerEncoder
 * Main encoder class that combines:
 * 1. Initial embedding/subsampling
 * 2. Pre-lookahead layer
 * 3. Multiple conformer encoder layers
 * 4. Upsampling layer
 * 5. Additional conformer encoder layers after upsampling
 * 6. Final normalization
 * 
 * Python equivalent:
 *   class UpsampleConformerEncoder(torch.nn.Module)
 */
struct UpsampleConformerEncoder {
    S3GenConfig config;
    
    // Embedding/subsampling layer weights
    ggml_tensor* embed_linear_weight;
    ggml_tensor* embed_linear_bias;
    ggml_tensor* embed_norm_weight;
    ggml_tensor* embed_norm_bias;
    
    // Positional encoding (for rel_pos_espnet, this is computed dynamically)
    ggml_tensor* pos_enc_pe;  // May be nullptr for certain pos_enc types
    
    // Pre-lookahead layer
    PreLookaheadLayer* pre_lookahead_layer;
    
    // Main encoder layers
    std::vector<ConformerEncoderLayer*> encoders;
    
    // Upsample layer
    Upsample1D* up_layer;
    
    // Upsampled embedding/subsampling layer weights
    ggml_tensor* up_embed_linear_weight;
    ggml_tensor* up_embed_linear_bias;
    ggml_tensor* up_embed_norm_weight;
    ggml_tensor* up_embed_norm_bias;

    int embed_d_model = 512;
    int embed_max_len = 5000;
    
    // Upsampled encoder layers
    std::vector<ConformerEncoderLayer*> up_encoders;
    
    // Final normalization
    ggml_tensor* after_norm_weight;
    ggml_tensor* after_norm_bias;
    
    // Constructor and destructor
    UpsampleConformerEncoder(const S3GenConfig& cfg = S3GenConfig::default_config());
    ~UpsampleConformerEncoder();
    
    // Model loading and management
    bool load_model(ggml_context* ggml_ctx);
    
    /**
     * Build computation graph for the encoder forward pass
     * @param ctx Compute context for building the graph
     * @param xs Input tensor (batch, time, input_size)
     * @param xs_lens Input lengths tensor (batch) - deprecated, use attention_mask
     * @param decoding_chunk_size Chunk size for streaming
     * @param num_decoding_left_chunks Number of left chunks for streaming
     * @param attention_mask Optional attention mask (1.0 for valid, 0.0 for padding)
     * @param upsampled_mask_out Output pointer for 2x upsampled attention mask
     * @return Output tensor (the final node in the graph)
     */
    ggml_tensor* build_graph(
        ggml_context* ctx,
        ggml_tensor* xs,
        ggml_tensor* xs_lens,
        int decoding_chunk_size = 0,
        int num_decoding_left_chunks = -1,
        ggml_tensor* attention_mask = nullptr,
        ggml_tensor** upsampled_mask_out = nullptr
    );

    // Forward through the main encoder layers
    ggml_tensor* forward_layers(ggml_context* ctx, ggml_tensor* xs,
        ggml_tensor* chunk_masks, ggml_tensor* pos_emb, ggml_tensor* mask_pad) {
        for (auto layer : encoders) {
            xs = layer->forward(ctx, xs, chunk_masks, pos_emb, mask_pad);
        }
        return xs;
    }

    // Forward through the upsampled encoder layers
    ggml_tensor* forward_up_layers(ggml_context* ctx, ggml_tensor* xs,
        ggml_tensor* chunk_masks, ggml_tensor* pos_emb, ggml_tensor* mask_pad) {
        for (auto layer : up_encoders) {
            xs = layer->forward(ctx, xs, chunk_masks, pos_emb, mask_pad);
        }
        return xs;
    }

    void print_model_info() const;
    int output_size() const { return config.encoder_output_size; }

private:
    ggml_tensor* checked_get_tensor(ggml_context* ctx, const char* name);
    void checked_validate_tensor(const ggml_tensor* tensor, const char* name);
    void print_shape(const char* name, const ggml_tensor* t) const;

    // Linear -> LayerNorm -> mask: embedding layer for encoder input
    ggml_tensor* apply_embed_layer(ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* linear_weight, ggml_tensor* linear_bias,
        ggml_tensor* norm_weight, ggml_tensor* norm_bias, ggml_tensor* mask) {
        ggml_tensor* out = ggml_add(ctx, ggml_mul_mat(ctx, linear_weight, x), linear_bias);
        out = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, out, 1e-5f), norm_weight), norm_bias);
        if (mask) out = ggml_mul(ctx, out, mask);
        return out;
    }
};


