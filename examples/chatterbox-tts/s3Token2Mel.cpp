#include "s3Token2Mel.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>



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
// S3Token2Mel Implementation
// ============================================================================

S3Token2Mel::S3Token2Mel(
    std::unique_ptr<UpsampleConformerEncoder> encoder_,
    std::unique_ptr<ConditionalDecoder> decoder_,
    const S3GenConfig& config_
) : config(config_),
    encoder(std::move(encoder_)),
    decoder(std::move(decoder_)),
    encoder_output_size(encoder ? encoder->config.encoder_output_size : config_.input_size),
    input_embedding_weight(nullptr),
    spk_embed_affine_weight(nullptr),
    spk_embed_affine_bias(nullptr),
    encoder_proj_weight(nullptr),
    encoder_proj_bias(nullptr) {
}

S3Token2Mel::~S3Token2Mel() {
}

bool S3Token2Mel::load_all(ggml_context* model_ctx) {
    printf("=== UpsampleConformerEncoder ===\n\n");
    if (!encoder || !encoder->load_model(model_ctx)) {
        fprintf(stderr, "ERROR: Failed to load UpsampleConformerEncoder\n");
        return false;
    }

    printf("=== ConditionalDecoder ===\n\n");
    if (!decoder || !decoder->load_model(model_ctx)) {
        fprintf(stderr, "ERROR: Failed to load ConditionalDecoder\n");
        return false;
    }

    printf("=== S3Token2Mel ===\n\n");
    if (!load_model(model_ctx)) {
        fprintf(stderr, "ERROR: Failed to load S3Token2Mel\n");
        return false;
    }

    return true;
}

void S3Token2Mel::set_rand_noise(ggml_tensor* noise) {
    if (decoder) {
        decoder->rand_noise = noise;
    }
}

void S3Token2Mel::clear_rand_noise() {
    if (decoder) {
        decoder->rand_noise = nullptr;
    }
}

bool S3Token2Mel::load_model(ggml_context* ggml_ctx)
{
    // Input embedding weight
    input_embedding_weight = ggml_get_tensor(ggml_ctx, "finput_emb_weight");
    CHECK_TENSOR_LOAD(input_embedding_weight, "finput_emb_weight");

    // Speaker embedding affine layer
    spk_embed_affine_weight = ggml_get_tensor(ggml_ctx, "fspk_affine_weight");
    CHECK_TENSOR_LOAD(spk_embed_affine_weight, "fspk_affine_weight");
    spk_embed_affine_bias = ggml_get_tensor(ggml_ctx, "fspk_affine_bias");
    CHECK_TENSOR_LOAD(spk_embed_affine_bias, "fspk_affine_bias");

    // Encoder projection layer
    encoder_proj_weight = ggml_get_tensor(ggml_ctx, "fenc_proj_weight");
    CHECK_TENSOR_LOAD(encoder_proj_weight, "fenc_proj_weight");
    encoder_proj_bias = ggml_get_tensor(ggml_ctx, "fenc_proj_bias");
    CHECK_TENSOR_LOAD(encoder_proj_bias, "fenc_proj_bias");
    
    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

ggml_tensor* S3Token2Mel::normalize_l2(
    ggml_context* ctx,
    ggml_tensor* x
) {
    // L2 normalize along dimension 1: x / ||x||_2
    // x shape: [batch, dim]
    // Compute norm: sqrt(sum(x^2))
    ggml_tensor* x_squared = ggml_mul(ctx, x, x);
    ggml_tensor* sum_squared = ggml_sum_rows(ctx, x_squared);
    ggml_tensor* norm = ggml_sqrt(ctx, sum_squared);
    
    // Avoid division by zero
    // norm = ggml_add1(ctx, norm, ggml_arange(ctx, 1e-12, 1, 1));
    norm = ggml_add(ctx, norm, ggml_arange(ctx, 1e-12, 1, 1));
    
    // Divide x by norm
    ggml_tensor* normalized = ggml_div(ctx, x, norm);
    
    return normalized;
}

ggml_tensor* S3Token2Mel::apply_encoder_projection(
    ggml_context* ctx,
    ggml_tensor* encoder_output
) {
    // encoder_output: [batch, seq_len, encoder_output_size]
    // output: [batch, seq_len, output_size]
    
    // Reshape for matrix multiplication
    int batch = static_cast<int>(encoder_output->ne[2]);
    int seq_len = static_cast<int>(encoder_output->ne[1]);
    
    // Reshape to [batch*seq_len, encoder_output_size]
    ggml_tensor* reshaped = ggml_reshape_2d(ctx, encoder_output,
                                            encoder_output_size,
                                            batch * seq_len);
    reshaped = ggml_cont(ctx, reshaped);
    
    // Apply linear layer
    ggml_tensor* projected = ggml_mul_mat(ctx, encoder_proj_weight, reshaped);
    projected = ggml_add(ctx, projected, encoder_proj_bias);
    
    // Reshape back to [batch, seq_len, output_size]
    projected = ggml_reshape_3d(ctx, projected, config.output_size, seq_len, batch);
    projected = ggml_cont(ctx, projected);
    
    return projected;
}

// ============================================================================
// Inference Function
// ============================================================================

std::pair<ggml_tensor*, ggml_tensor*> S3Token2Mel::inference(
    ggml_context* ctx,
    ggml_tensor* token,
    ggml_tensor* prompt_token,
    ggml_tensor* prompt_feat,
    ggml_tensor* embedding,
    ggml_tensor* attention_mask,
    bool finalize
) {
    int token_len = static_cast<int>(token->ne[0]);
    int prompt_token_len = static_cast<int>(prompt_token->ne[0]);
    int prompt_feat_len = static_cast<int>(prompt_feat->ne[1]);
    // Step 1: Normalize and project speaker embedding
    // embedding = F.normalize(embedding, dim=1)
    ggml_tensor* norm_embedding = normalize_l2(ctx, embedding);
    
    // embedding = self.spk_embed_affine_layer(embedding)
    ggml_tensor* proj_embedding = apply_spk_embed_projection(ctx, norm_embedding);
    ggml_set_name(proj_embedding, "dbg_spk_proj");
    ggml_set_output(proj_embedding);

    // Step 2: Concatenate prompt_token and token
    // token = torch.concat([prompt_token, token], dim=1)
    ggml_tensor* concat_token = ggml_concat(ctx, prompt_token, token, 0);
    int total_token_len = static_cast<int>(concat_token->ne[0]);
    
    // Step 3: Prepare mask for token embedding
    // If attention_mask provided, use it; otherwise create from lengths
    ggml_tensor* token_len_tensor = nullptr;
    ggml_tensor* mask_3d = nullptr;
    
    if (attention_mask) {
        // Reshape attention_mask from (total_len, 1) to (1, total_len, 1) for broadcasting
        // mask values: 1.0 = valid, 0.0 = padded
        mask_3d = ggml_reshape_3d(ctx, attention_mask, 1, total_token_len, 1);
    } else {
        // Fall back to computing mask from lengths - only create tensor when needed
        token_len_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* mask = make_pad_mask(ctx, token_len_tensor, total_token_len);
        mask_3d = ggml_reshape_3d(ctx, mask, 1, total_token_len, 1);
    }
    
    // Step 4: Clamp tokens and apply embedding
    // token = self.input_embedding(torch.clamp(token, min=0, max=vocab_size-1)) * mask
    ggml_tensor* clamped_tokens = clamp_tokens(ctx, concat_token);
    ggml_tensor* token_embeds = apply_input_embedding(ctx, clamped_tokens);
    // Apply mask to token embeddings: zeros out padding positions
    // token_embeds shape: (total_token_len, input_size, 1)
    // mask_3d shape: (total_token_len, 1, 1) - will broadcast
    token_embeds = ggml_mul(ctx, token_embeds, mask_3d);
    ggml_set_name(token_embeds, "dbg_tok_emb");
    ggml_set_output(token_embeds);
    // Step 5: Encoder forward pass with attention mask
    // h, h_lengths = self.encoder(token, token_len)
    ggml_tensor* h = nullptr;
    ggml_tensor* upsampled_mask = nullptr;
    
    if (encoder) {
        h = encoder->build_graph(ctx, token_embeds, token_len_tensor, 0, -1, 
                                  attention_mask, &upsampled_mask);
        
    } else {
        fprintf(stderr, "  ERROR: No encoder provided\n");
        return { nullptr, nullptr };
    }
    // Step 6: Apply lookahead trimming if not finalizing
    // if finalize is False:
    //     h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]
    if (!finalize) {
        int trim_len = config.pre_lookahead_len * config.token_mel_ratio;
        int new_seq_len = static_cast<int>(h->ne[1]) - trim_len;
        
        if (new_seq_len > 0) {
            h = ggml_view_3d(ctx, h,
                            h->ne[0], new_seq_len, h->ne[2],
                            h->nb[1], h->nb[2], 0);
        }
    }
    
    // Step 7: Calculate mel lengths
    // mel_len1 = prompt_feat.shape[1]
    // mel_len2 = h.shape[1] - prompt_feat.shape[1]
    int mel_len1 = prompt_feat_len;
    int mel_len2 = static_cast<int>(h->ne[1]) - prompt_feat_len;
    
    // Step 8: Apply encoder projection
    // h = self.encoder_proj(h)
    h = apply_encoder_projection(ctx, h);
    
    // Step 9: Prepare conditions
    // NOTE: The conds tensor dimensions depend on mel_len1 (from prompt_feat_frames) and mel_len2.
    // For graph caching to work correctly, S3Token2Wav::needs_graph_rebuild() checks that
    // prompt_feat_frames matches the cached value - if it differs, the graph is rebuilt.
    int total_mel_len = mel_len1 + mel_len2;
    ggml_tensor* conds = nullptr;
    if (mel_len1 > 0 && prompt_feat) {
        conds = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
            config.output_size, mel_len2, 1);
        conds = ggml_repeat(ctx, ggml_arange(ctx, 0, 1, 1), conds);
    
        // Copy prompt features to first mel_len1 positions
        conds = ggml_concat(ctx, prompt_feat, conds, 1);
    }
    else
    {
        conds = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
            config.output_size, total_mel_len, 1);
        conds = ggml_repeat(ctx, ggml_arange(ctx, 0, 1, 1), conds);
    }
    
    // Transpose: [mel_len, output_size, 1] -> [output_size, mel_len, 1]
    conds = ggml_cont(ctx, ggml_permute(ctx, conds, 1, 0, 2, 3));
    
    // Step 10: Create mask for decoder
    // Use upsampled_mask from encoder if available, otherwise compute from lengths
    ggml_tensor* decoder_mask = nullptr;
    if (upsampled_mask) {
        // upsampled_mask shape: (1, upsampled_len) -> reshape to (upsampled_len, 1, 1)
        decoder_mask = ggml_reshape_3d(ctx, upsampled_mask, total_mel_len, 1, 1);
    } else {
        // Fall back to computing mask from lengths
        ggml_tensor* mel_len_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        decoder_mask = make_pad_mask(ctx, mel_len_tensor, total_mel_len);
        decoder_mask = ggml_reshape_3d(ctx, decoder_mask, total_mel_len, 1, 1);
    }
    
    // Step 11: Transpose h for decoder
    // h.transpose(1, 2).contiguous()
    h = ggml_permute(ctx, h, 1, 0, 2, 3);
    h = ggml_cont(ctx, h);

    // Step 12: Call decoder
    // feat, _ = self.decoder(mu=h, mask=mask, spks=embedding, cond=conds, n_timesteps=n_timesteps)
    if (!decoder) {
        fprintf(stderr, "ERROR: Decoder is null!\n");
        return std::make_pair((ggml_tensor*)nullptr, (ggml_tensor*)nullptr);
    }
    // Use n_timesteps from config: meanflow=2, non-turbo=5
    int n_timesteps = config.n_timesteps;
    auto result = decoder->forward_causal(
        ctx,
        h,                          // mu
        decoder_mask,               // mask
        n_timesteps,                // n_timesteps from config
        1.0f,                       // temperature
        proj_embedding,             // spks
        conds                       // cond
    );
    // return{result.first, nullptr};

    ggml_tensor* feat = result.first;
    //return { feat , h };
     //Step 13: Extract only the new generated portion
     //feat = feat[:, :, mel_len1:]
     if (mel_len1 > 0 && feat->ne[0] > mel_len1) {
         feat = ggml_view_3d(ctx, feat,
                            feat->ne[0] - mel_len1, feat->ne[1], feat->ne[2],
                            feat->nb[1], feat->nb[2],
                            mel_len1 * feat->nb[0]);
     }
    
     // Verify output size matches expected
     if (feat->ne[0] != mel_len2) {
         fprintf(stderr, "WARNING: Output mel length mismatch! Expected %d, got %d\n",
                 mel_len2, (int)feat->ne[0]);
     }
    
     feat = ggml_cont(ctx, feat);
     // Return (feat, None) - no flow cache for causal mode
     return std::make_pair(feat, (ggml_tensor*)nullptr);
}
