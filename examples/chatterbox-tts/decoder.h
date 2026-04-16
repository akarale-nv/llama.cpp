#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include "s3gen_config.h"
#include "s3gen_ops.h"

// Forward declarations
struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;

// ============================================================================
// Attention / Transformer blocks
// ============================================================================

struct BasicTransformerBlock {
    int dim;
    int num_attention_heads;
    int attention_head_dim;
    float dropout;
    std::string activation_fn;

    ggml_tensor* norm1_weight;
    ggml_tensor* norm1_bias;
    ggml_tensor* norm2_weight;
    ggml_tensor* norm2_bias;
    ggml_tensor* norm3_weight;
    ggml_tensor* norm3_bias;

    ggml_tensor* attn1_to_q_weight;
    ggml_tensor* attn1_to_k_weight;
    ggml_tensor* attn1_to_v_weight;
    ggml_tensor* attn1_to_out_weight;
    ggml_tensor* attn1_to_out_bias;

    ggml_tensor* ff_net_0_proj_weight;
    ggml_tensor* ff_net_0_proj_bias;
    ggml_tensor* ff_net_2_weight;
    ggml_tensor* ff_net_2_bias;

    BasicTransformerBlock(int dim_, int num_attention_heads_, int attention_head_dim_,
        float dropout_ = 0.0f, const char* activation_fn_ = "gelu");
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* hidden_states,
        ggml_tensor* attention_mask, ggml_tensor* timestep = nullptr,
        bool use_flash_attention = true);
    bool load_weights(ggml_context* model_ctx, const char* prefix);
};

// ============================================================================
// Building blocks (from matcha_decoder)
// ============================================================================

struct TimestepEmbedding {
    int in_channels;
    int time_embed_dim;
    std::string act_fn;

    ggml_tensor* linear_1_weight;
    ggml_tensor* linear_1_bias;
    ggml_tensor* linear_2_weight;
    ggml_tensor* linear_2_bias;

    TimestepEmbedding(int in_channels_, int time_embed_dim_, const char* act_fn_ = "silu");
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* sample);
    bool load_weights(ggml_context* model_ctx, const char* prefix);
};

struct Block1D {
    int dim;
    int dim_out;
    int groups;
    bool causal;

    ggml_tensor* conv_weight;
    ggml_tensor* conv_bias;
    ggml_tensor* norm_weight;
    ggml_tensor* norm_bias;

    Block1D(int dim_, int dim_out_, int groups_ = 8, bool causal_ = false);
    ~Block1D() = default;
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask);
    bool load_weights(ggml_context* model_ctx, const char* prefix);
};

struct ResnetBlock1D {
    int dim;
    int dim_out;
    int time_emb_dim;
    int groups;
    bool causal;

    ggml_tensor* mlp_weight;
    ggml_tensor* mlp_bias;
    std::unique_ptr<Block1D> block1;
    std::unique_ptr<Block1D> block2;
    ggml_tensor* res_conv_weight;
    ggml_tensor* res_conv_bias;

    ResnetBlock1D(int dim_, int dim_out_, int time_emb_dim_, int groups_ = 8, bool causal_ = false);
    ~ResnetBlock1D() = default;
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask, ggml_tensor* time_emb);
    bool load_weights(ggml_context* model_ctx, const char* prefix);
};

struct Downsample1D {
    int dim;
    ggml_tensor* conv_weight;
    ggml_tensor* conv_bias;

    Downsample1D(int dim_);
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
    bool load_weights(ggml_context* model_ctx, const char* prefix);
};

struct MatchaUpsample1D {
    int channels;
    int out_channels;
    bool use_conv_transpose;
    bool use_conv;

    ggml_tensor* conv_weight;
    ggml_tensor* conv_bias;

    MatchaUpsample1D(int channels_, int out_channels_, bool use_conv_transpose_, bool use_conv);
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x);
    bool load_weights(ggml_context* model_ctx, const char* prefix);
};

/**
 * Conditional Decoder (UNet-style architecture)
 * Main estimator network for flow matching
 * 
 * Python equivalent:
 *   class ConditionalDecoder(nn.Module)
 */
struct ConditionalDecoder {
    S3GenConfig config;
    
    // Time embeddings - using unique_ptr for RAII
    int time_emb_dim_sinusoidal = 0;  // dim for sinusoidal_pos_emb()
    std::unique_ptr<TimestepEmbedding> time_mlp;
    ggml_tensor* time_embed_mixer_weight = nullptr;  // For meanflow: mixes t_emb and r_emb
    
    // Down blocks: [resnet, transformer_blocks, downsample]
    struct DownBlock {
        std::unique_ptr<ResnetBlock1D> resnet;
        std::vector<std::unique_ptr<BasicTransformerBlock>> transformer_blocks;
        ggml_tensor* downsample_weight;  // Either Downsample1D or Conv1d
        ggml_tensor* downsample_bias;
        bool is_downsample;  // true if Downsample1D, false if Conv1d
    };
    std::vector<DownBlock> down_blocks;
    
    // Mid blocks: [resnet, transformer_blocks]
    struct MidBlock {
        std::unique_ptr<ResnetBlock1D> resnet;
        std::vector<std::unique_ptr<BasicTransformerBlock>> transformer_blocks;
    };
    std::vector<MidBlock> mid_blocks;
    
    // Up blocks: [resnet, transformer_blocks, upsample]
    struct UpBlock {
        std::unique_ptr<ResnetBlock1D> resnet;
        std::vector<std::unique_ptr<BasicTransformerBlock>> transformer_blocks;
        std::unique_ptr<MatchaUpsample1D> upsample;  // Either MatchaUpsample1D or Conv1d
        ggml_tensor* upsample_conv_weight;  // For Conv1d case
        ggml_tensor* upsample_conv_bias;
        bool is_upsample;  // true if MatchaUpsample1D , false if Conv1d
    };
    std::vector<UpBlock> up_blocks;
    
    // Final layers - using unique_ptr for RAII
    std::unique_ptr<Block1D> final_block;
    ggml_tensor* final_proj_weight = nullptr;  // (out_channels, channels[-1], 1)
    ggml_tensor* final_proj_bias = nullptr;    // (out_channels)
    
    // Constructor (destructor not needed - unique_ptr handles cleanup)
    ConditionalDecoder(const S3GenConfig& cfg);
    ~ConditionalDecoder() = default;
    
    // Model loading and management
    //bool load_model(const char* model_path);
    bool load_model(ggml_context* ggml_ctx);
    
    /**
     * Forward pass of the UNet1DConditional model
     * 
     * @param ctx Compute context for building the graph
     * @param x Input tensor (batch_size, in_channels, time)
     * @param mask Mask tensor (batch_size, 1, time)
     * @param mu Encoder output tensor (batch_size, mu_channels, time)
     * @param t Timestep tensor (batch_size,)
     * @param spks Speaker embedding tensor (batch_size, spk_emb_dim) - optional
     * @param cond Conditional input tensor (batch_size, cond_channels, time) - optional
     * @return Output tensor (batch_size, out_channels, time)
     */
    ggml_tensor* forward(
        ggml_context* ctx,
        ggml_tensor* x,
        ggml_tensor* mask,
        ggml_tensor* mu,
        ggml_tensor* t,
        ggml_tensor* spks = nullptr,
        ggml_tensor* cond = nullptr,
        ggml_tensor* r = nullptr
    );
    
    void print_model_info() const;

    // === Flow matching (absorbed from CausalConditionalCFM) ===
    ggml_tensor* rand_noise = nullptr;  // Pre-allocated random noise [1, 80, 50*300]

    std::pair<ggml_tensor*, ggml_tensor*> forward_causal(
        ggml_context* ctx, ggml_tensor* mu, ggml_tensor* mask,
        int n_timesteps, float temperature,
        ggml_tensor* spks, ggml_tensor* cond);

    ggml_tensor* solve_euler(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* t_span,
        ggml_tensor* mu, ggml_tensor* mask,
        ggml_tensor* spks, ggml_tensor* cond);

    ggml_tensor* basic_euler(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* time_span,
        ggml_tensor* mu, ggml_tensor* mask,
        ggml_tensor* spks, ggml_tensor* cond);

private:
    // Helper functions
    ggml_tensor* checked_get_tensor(ggml_context* ctx, const char* name);
    void print_shape(const char* name, const ggml_tensor* t) const;
    
    // Helper: Create attention mask with optional chunking
    ggml_tensor* create_attention_mask(
        ggml_context* ctx,
        ggml_tensor* mask,
        bool causal = false
    );
};

