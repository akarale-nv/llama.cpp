#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "s3gen_config.h"
#include "decoder.h"
#include "encoder.h"
#include <vector>
#include <string>
#include <cstdint>
#include <memory>

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ConditionalDecoder;
struct UpsampleConformerEncoder;

/**
 * S3Token2Mel
 * 
 * A causal flow-based TTS model that:
 * 1. Takes token input and encodes it
 * 2. Projects encoded features to mel space
 * 3. Uses speaker embeddings for voice control
 * 4. Generates mel-spectrograms using causal flow matching
 * 
 * This is designed for streaming/online inference where tokens are
 * processed causally without access to future context.
 */
struct S3Token2Mel {
    // Configuration
    S3GenConfig config;
    
    // Model components (weights)
    ggml_tensor* input_embedding_weight;    // [vocab_size, input_size]
    ggml_tensor* spk_embed_affine_weight;   // [spk_embed_dim, output_size]
    ggml_tensor* spk_embed_affine_bias;     // [output_size]
    ggml_tensor* encoder_proj_weight;       // [encoder_output_size, output_size]
    ggml_tensor* encoder_proj_bias;         // [output_size]
    
    // Owned sub-components
    std::unique_ptr<UpsampleConformerEncoder> encoder;
    int encoder_output_size;
    std::unique_ptr<ConditionalDecoder> decoder;

    /**
     * Constructor — takes ownership of encoder and decoder.
     */
    S3Token2Mel(
        std::unique_ptr<UpsampleConformerEncoder> encoder_,
        std::unique_ptr<ConditionalDecoder> decoder_,
        const S3GenConfig& config_ = S3GenConfig::default_config()
    );

    ~S3Token2Mel();

    /**
     * Load weights for all sub-models (encoder, decoder, and flow projection).
     */
    bool load_all(ggml_context* model_ctx);

    /**
     * Load only the flow projection weights (input embedding, spk affine, encoder proj).
     */
    bool load_model(ggml_context* model_ctx);

    // Manage decoder rand_noise from the outside (for graph caching)
    void set_rand_noise(ggml_tensor* noise);
    void clear_rand_noise();

    /**
     * @param ctx GGML context for computation graph
     * @param token Token IDs tensor
     * @param prompt_token Prompt token IDs tensor
     * @param prompt_feat Prompt mel features tensor
     * @param embedding Speaker embedding tensor
     * @param attention_mask Attention mask tensor (1.0 for valid, 0.0 for padding)
     * @param finalize Whether this is the final chunk (no lookahead trimming)
     * @return Pair of (generated mel features, flow_cache)
     */
    std::pair<ggml_tensor*, ggml_tensor*> inference(
        ggml_context* ctx,
        ggml_tensor* token,
        ggml_tensor* prompt_token,
        ggml_tensor* prompt_feat,
        ggml_tensor* embedding,
        ggml_tensor* attention_mask,
        bool finalize = true
    );
    
    // Embedding lookup: token_ids [batch, seq_len] -> [batch, seq_len, input_size]
    ggml_tensor* apply_input_embedding(ggml_context* ctx, ggml_tensor* token_ids) {
        return ggml_get_rows(ctx, input_embedding_weight, token_ids);
    }

    // Speaker embedding projection: [batch, spk_embed_dim] -> [batch, output_size]
    ggml_tensor* apply_spk_embed_projection(ggml_context* ctx, ggml_tensor* embedding) {
        return ggml_add(ctx, ggml_mul_mat(ctx, spk_embed_affine_weight, embedding),
                        spk_embed_affine_bias);
    }

    // Encoder output projection: [batch, seq_len, encoder_output_size] -> [batch, seq_len, output_size]
    ggml_tensor* apply_encoder_projection(ggml_context* ctx, ggml_tensor* encoder_output);

    // L2 normalize along dim 1: x / ||x||_2
    ggml_tensor* normalize_l2(ggml_context* ctx, ggml_tensor* x);

    // Token clamping (currently a no-op, model handles valid ranges)
    ggml_tensor* clamp_tokens([[maybe_unused]] ggml_context* ctx, ggml_tensor* tokens) {
        return tokens;
    }
};
