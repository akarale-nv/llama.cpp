#include "decoder.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <memory>


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
// TimestepEmbedding Implementation
// ============================================================================

TimestepEmbedding::TimestepEmbedding(int in_channels_, int time_embed_dim_, const char* act_fn_)
    : in_channels(in_channels_), time_embed_dim(time_embed_dim_), act_fn(act_fn_),
      linear_1_weight(nullptr), linear_1_bias(nullptr),
      linear_2_weight(nullptr), linear_2_bias(nullptr) {
}

ggml_tensor* TimestepEmbedding::forward(ggml_context* ctx, ggml_tensor* sample) {
    ggml_tensor* x = ggml_mul_mat(ctx, linear_1_weight, sample);
    x = ggml_add(ctx, x, linear_1_bias);
    if (act_fn == "silu") {
        x = apply_silu(ctx, x);
    } else if (act_fn == "gelu") {
        x = ggml_gelu(ctx, x);
    }
    x = ggml_mul_mat(ctx, linear_2_weight, x);
    x = ggml_add(ctx, x, linear_2_bias);
    return x;
}

bool TimestepEmbedding::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s_linear_1_weight", prefix);
    linear_1_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(linear_1_weight, name);
    snprintf(name, sizeof(name), "%s_linear_1_bias", prefix);
    linear_1_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(linear_1_bias, name);
    snprintf(name, sizeof(name), "%s_linear_2_weight", prefix);
    linear_2_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(linear_2_weight, name);
    snprintf(name, sizeof(name), "%s_linear_2_bias", prefix);
    linear_2_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(linear_2_bias, name);
    return true;
}

// ============================================================================
// Block1D Implementation
// ============================================================================

Block1D::Block1D(int dim_, int dim_out_, int groups_, bool causal_)
    : dim(dim_), dim_out(dim_out_), groups(groups_), causal(causal_),
      conv_weight(nullptr), conv_bias(nullptr),
      norm_weight(nullptr), norm_bias(nullptr) {
}

ggml_tensor* Block1D::forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask) {
    x = ggml_mul(ctx, x, mask);
    if (causal) {
        x = apply_causal_conv1d(ctx, x, conv_weight, conv_bias, 3);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = apply_layer_norm(ctx, x, norm_weight, norm_bias, 1e-5f);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
    } else {
        x = apply_conv1d(ctx, x, conv_weight, conv_bias, 1, 1, 1, static_cast<int>(x->ne[2]));
        x = ggml_group_norm(ctx, x, groups, 1e-5f);
        ggml_tensor* norm_weight_reshape = ggml_reshape_2d(ctx, norm_weight, 1, norm_weight->ne[0]);
        x = ggml_mul(ctx, x, norm_weight_reshape);
        if (norm_bias) {
            ggml_tensor* norm_bias_reshape = ggml_reshape_2d(ctx, norm_bias, 1, norm_bias->ne[0]);
            x = ggml_add(ctx, x, norm_bias_reshape);
        }
    }
    x = apply_mish(ctx, x);
    x = ggml_mul(ctx, x, mask);
    return x;
}

bool Block1D::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s_blk_0_weight", prefix);
    conv_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(conv_weight, name);
    snprintf(name, sizeof(name), "%s_blk_0_bias", prefix);
    conv_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(conv_bias, name);
    snprintf(name, sizeof(name), "%s_blk_2_weight", prefix);
    norm_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(norm_weight, name);
    snprintf(name, sizeof(name), "%s_blk_2_bias", prefix);
    norm_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(norm_bias, name);
    return true;
}

// ============================================================================
// ResnetBlock1D Implementation
// ============================================================================

ResnetBlock1D::ResnetBlock1D(int dim_, int dim_out_, int time_emb_dim_, int groups_, bool causal_)
    : dim(dim_), dim_out(dim_out_), time_emb_dim(time_emb_dim_), groups(groups_), causal(causal_),
      mlp_weight(nullptr), mlp_bias(nullptr),
      res_conv_weight(nullptr), res_conv_bias(nullptr) {
    block1 = std::make_unique<Block1D>(dim, dim_out, groups, causal);
    block2 = std::make_unique<Block1D>(dim_out, dim_out, groups, causal);
}

ggml_tensor* ResnetBlock1D::forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask, ggml_tensor* time_emb) {
    ggml_tensor* residual = ggml_mul(ctx, x, mask);
    ggml_tensor* h = block1->forward(ctx, x, mask);
    ggml_tensor* time_proj = apply_mish(ctx, time_emb);
    time_proj = ggml_mul_mat(ctx, mlp_weight, time_proj);
    time_proj = ggml_add(ctx, time_proj, mlp_bias);
    time_proj = ggml_reshape_3d(ctx, time_proj, 1, time_proj->ne[0], time_proj->ne[1]);
    h = ggml_add(ctx, h, time_proj);
    h = block2->forward(ctx, h, mask);
    ggml_tensor* res_conv_weight_reshape = ggml_reshape_3d(ctx, res_conv_weight, 1, res_conv_weight->ne[0], res_conv_weight->ne[1]);
    residual = apply_conv1d(ctx, residual, res_conv_weight_reshape, res_conv_bias, 1, 0, 1, static_cast<int>(residual->ne[2]));
    return ggml_add(ctx, h, residual);
}

bool ResnetBlock1D::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s_mlp_1_weight", prefix);
    mlp_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(mlp_weight, name);
    snprintf(name, sizeof(name), "%s_mlp_1_bias", prefix);
    mlp_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(mlp_bias, name);
    snprintf(name, sizeof(name), "%s_res_conv_weight", prefix);
    res_conv_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(res_conv_weight, name);
    snprintf(name, sizeof(name), "%s_res_conv_bias", prefix);
    res_conv_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(res_conv_bias, name);
    snprintf(name, sizeof(name), "%s_blk1", prefix);
    if (block1 && !block1->load_weights(model_ctx, name)) return false;
    snprintf(name, sizeof(name), "%s_blk2", prefix);
    if (block2 && !block2->load_weights(model_ctx, name)) return false;
    return true;
}

// ============================================================================
// Downsample1D Implementation
// ============================================================================

Downsample1D::Downsample1D(int dim_)
    : dim(dim_), conv_weight(nullptr), conv_bias(nullptr) {
}

ggml_tensor* Downsample1D::forward(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* w = (conv_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv_weight, GGML_TYPE_F32) : conv_weight;
    return ggml_conv_1d(ctx, w, x, 2, 1, 1);
}

bool Downsample1D::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s_conv_weight", prefix);
    conv_weight = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(conv_weight, name);
    snprintf(name, sizeof(name), "%s_conv_bias", prefix);
    conv_bias = ggml_get_tensor(model_ctx, name);
    CHECK_TENSOR_LOAD(conv_bias, name);
    return true;
}

// ============================================================================
// MatchaUpsample1D Implementation
// ============================================================================

MatchaUpsample1D::MatchaUpsample1D(int channels_, int out_channels_, bool use_conv_transpose_, bool use_conv_)
    : channels(channels_), out_channels(out_channels_), use_conv_transpose(use_conv_transpose_), use_conv(use_conv_),
      conv_weight(nullptr), conv_bias(nullptr) {
}

ggml_tensor* MatchaUpsample1D::forward(ggml_context* ctx, ggml_tensor* x) {
    if (use_conv_transpose) {
        return ggml_conv_transpose_1d(ctx, conv_weight, x, 4, 2, 1);
    } else {
        x = ggml_upscale_ext(ctx, x, static_cast<int>(x->ne[0]) * 2, static_cast<int>(x->ne[1]), static_cast<int>(x->ne[2]), static_cast<int>(x->ne[3]), GGML_SCALE_MODE_NEAREST);
        if (use_conv) {
            ggml_tensor* w = (conv_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, conv_weight, GGML_TYPE_F32) : conv_weight;
            x = ggml_conv_1d(ctx, w, x, 3, 1, 1);
            if (conv_bias != nullptr) {
                auto* conv_bias_reshape = ggml_cont(ctx, ggml_transpose(ctx, conv_bias));
                x = ggml_add(ctx, x, conv_bias_reshape);
            }
        }
        return x;
    }
}

bool MatchaUpsample1D::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name[256];
    if (use_conv_transpose || use_conv) {
        snprintf(name, sizeof(name), "%s_conv_weight", prefix);
        conv_weight = ggml_get_tensor(model_ctx, name);
        CHECK_TENSOR_LOAD(conv_weight, name);
        snprintf(name, sizeof(name), "%s_conv_bias", prefix);
        conv_bias = ggml_get_tensor(model_ctx, name);
        CHECK_TENSOR_LOAD(conv_bias, name);
    }
    return true;
}

// ============================================================================
// ConditionalDecoder Implementation
// ============================================================================

ConditionalDecoder::ConditionalDecoder(const S3GenConfig& cfg)
    : config(cfg) {
    // Note: time_embed_mixer_weight, final_proj_weight, final_proj_bias use in-class initializers

    // Initialize time embeddings using make_unique for RAII
    time_emb_dim_sinusoidal = cfg.decoder_in_channels;
    int time_embed_dim = cfg.channels[0] * 4;
    time_mlp = std::make_unique<TimestepEmbedding>(cfg.decoder_in_channels, time_embed_dim, "silu");

    // Initialize down blocks
    int output_channel = cfg.decoder_in_channels;
    for (size_t i = 0; i < cfg.channels.size(); i++) {
        int input_channel = output_channel;
        output_channel = cfg.channels[i];
        bool is_last = (i == cfg.channels.size() - 1);
        
        DownBlock down_block;
        down_block.resnet = std::make_unique<ResnetBlock1D>(input_channel, output_channel, time_embed_dim, 8, /*causal=*/true);
        
        // Create transformer blocks using make_unique
        for (int j = 0; j < cfg.n_blocks; j++) {
            down_block.transformer_blocks.push_back(std::make_unique<BasicTransformerBlock>(
                output_channel,
                cfg.num_heads,
                cfg.attention_head_dim,
                cfg.dropout,
                cfg.act_fn.c_str()
            ));
        }
        
        down_block.is_downsample = !is_last;
        
        down_blocks.push_back(std::move(down_block));
    }
    
    // Initialize mid blocks
    for (int i = 0; i < cfg.num_mid_blocks; i++) {
        MidBlock mid_block;
        mid_block.resnet = std::make_unique<ResnetBlock1D>(cfg.channels.back(), cfg.channels.back(), time_embed_dim, 8, /*causal=*/true);
        
        // Create transformer blocks using make_unique
        for (int j = 0; j < cfg.n_blocks; j++) {
            mid_block.transformer_blocks.push_back(std::make_unique<BasicTransformerBlock>(
                cfg.channels.back(),
                cfg.num_heads,
                cfg.attention_head_dim,
                cfg.dropout,
                cfg.act_fn.c_str()
            ));
        }
        
        mid_blocks.push_back(std::move(mid_block));
    }
    
    // Initialize up blocks
    std::vector<int> up_channels = cfg.channels;
    std::reverse(up_channels.begin(), up_channels.end());
    up_channels.push_back(cfg.channels[0]);
    
    for (size_t i = 0; i < up_channels.size() - 1; i++) {
        int input_channel = up_channels[i] * 2;  // Account for skip connection
        int output_channel = up_channels[i + 1];
        bool is_last = (i == up_channels.size() - 2);
        
        UpBlock up_block;
        up_block.resnet = std::make_unique<ResnetBlock1D>(input_channel, output_channel, time_embed_dim, 8, /*causal=*/true);
        
        // Create transformer blocks using make_unique
        for (int j = 0; j < cfg.n_blocks; j++) {
            up_block.transformer_blocks.push_back(std::make_unique<BasicTransformerBlock>(
                output_channel,
                cfg.num_heads,
                cfg.attention_head_dim,
                cfg.dropout,
                cfg.act_fn.c_str()
            ));
        }
        
        up_block.is_upsample = !is_last;
        if (!is_last) {
            up_block.upsample = std::make_unique<MatchaUpsample1D>(output_channel, output_channel, true, false);
        }
        
        up_blocks.push_back(std::move(up_block));
    }
    
    // Initialize final layers using make_unique
    final_block = std::make_unique<Block1D>(cfg.channels[0], cfg.channels[0], 8, cfg.causal);
}

// Destructor is defaulted - unique_ptr handles cleanup automatically

// Helper: Load transformer blocks for a given block
static bool load_transformer_blocks(
    ggml_context* ggml_ctx,
    const std::vector<std::unique_ptr<BasicTransformerBlock>>& blocks,
    const char* block_type,
    size_t block_idx
) {
    char prefix[256];
    for (size_t j = 0; j < blocks.size(); j++) {
        snprintf(prefix, sizeof(prefix), "fdes_%s_%zu_1_%zu", block_type, block_idx, j);
        if (!blocks[j]->load_weights(ggml_ctx, prefix)) {
            fprintf(stderr, "Failed to load %s\n", prefix);
            return false;
        }
    }
    return true;
}

// Helper: Load resnet for a block (accepts unique_ptr's underlying pointer)
static bool load_block_resnet(
    ggml_context* ggml_ctx,
    ResnetBlock1D* resnet,
    const char* block_type,
    size_t block_idx
) {
    if (!resnet) return true;
    
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "fdes_%s_%zu_0", block_type, block_idx);
    if (!resnet->load_weights(ggml_ctx, prefix)) {
        fprintf(stderr, "Failed to load %s\n", prefix);
        return false;
    }
    return true;
}

bool ConditionalDecoder::load_model(ggml_context* ggml_ctx)
{
    fprintf(stdout, "Loading time_mlp weights...\n");
    if (time_mlp && !time_mlp->load_weights(ggml_ctx, "fdes_time_mlp")) {
        fprintf(stderr, "Failed to load fdes_time_mlp");
        return false;
    }

    if (config.meanflow) {
        fprintf(stdout, "Loading time_embed_mixer weights...\n");
        time_embed_mixer_weight = ggml_get_tensor(ggml_ctx, "fdes_time_embed_mixer_weight");
        CHECK_TENSOR_LOAD(time_embed_mixer_weight, "fdes_time_embed_mixer_weight");
    }

    // Load down blocks
    fprintf(stderr, "Loading %zu down blocks...\n", down_blocks.size());
    for (size_t i = 0; i < down_blocks.size(); i++) {
        if (!load_block_resnet(ggml_ctx, down_blocks[i].resnet.get(), "down_blks", i)) return false;
        if (!load_transformer_blocks(ggml_ctx, down_blocks[i].transformer_blocks, "down_blks", i)) return false;

        char prefix[256];
        snprintf(prefix, sizeof(prefix), "fdes_down_blks_%zu_2_weight", i);
        down_blocks[i].downsample_weight = ggml_get_tensor(ggml_ctx, prefix);
        CHECK_TENSOR_LOAD(down_blocks[i].downsample_weight, prefix);

        snprintf(prefix, sizeof(prefix), "fdes_down_blks_%zu_2_bias", i);
        down_blocks[i].downsample_bias = ggml_get_tensor(ggml_ctx, prefix);
        CHECK_TENSOR_LOAD(down_blocks[i].downsample_bias, prefix);
    }

    // Load mid blocks
    fprintf(stderr, "Loading %zu mid blocks...\n", mid_blocks.size());
    for (size_t i = 0; i < mid_blocks.size(); i++) {
        if (!load_block_resnet(ggml_ctx, mid_blocks[i].resnet.get(), "mid_blks", i)) return false;
        if (!load_transformer_blocks(ggml_ctx, mid_blocks[i].transformer_blocks, "mid_blks", i)) return false;
    }

    // Load up blocks
    fprintf(stderr, "Loading %zu up blocks...\n", up_blocks.size());
    for (size_t i = 0; i < up_blocks.size(); i++) {
        if (!load_block_resnet(ggml_ctx, up_blocks[i].resnet.get(), "up_blks", i)) return false;
        if (!load_transformer_blocks(ggml_ctx, up_blocks[i].transformer_blocks, "up_blks", i)) return false;

        char prefix[256];
        if (up_blocks[i].upsample) {
            snprintf(prefix, sizeof(prefix), "fdes_up_blks_%zu_2", i);
            if (!up_blocks[i].upsample->load_weights(ggml_ctx, prefix)) {
                fprintf(stderr, "Failed to load %s\n", prefix);
                return false;
            }
        } else {
            snprintf(prefix, sizeof(prefix), "fdes_up_blks_%zu_2_weight", i);
            up_blocks[i].upsample_conv_weight = ggml_get_tensor(ggml_ctx, prefix);
            CHECK_TENSOR_LOAD(up_blocks[i].upsample_conv_weight, prefix);

            snprintf(prefix, sizeof(prefix), "fdes_up_blks_%zu_2_bias", i);
            up_blocks[i].upsample_conv_bias = ggml_get_tensor(ggml_ctx, prefix);
            CHECK_TENSOR_LOAD(up_blocks[i].upsample_conv_bias, prefix);
        }
    }

    // Load final block
    fprintf(stderr, "Loading final_block weights...\n");
    if (final_block && !final_block->load_weights(ggml_ctx, "fdes_final_blk")) {
        fprintf(stderr, "Failed to load fdes_final_blk");
        return false;
    }

    // Load final projection
    final_proj_weight = ggml_get_tensor(ggml_ctx, "fdes_final_proj_weight");
    CHECK_TENSOR_LOAD(final_proj_weight, "fdes_final_proj_weight");

    final_proj_bias = ggml_get_tensor(ggml_ctx, "fdes_final_proj_bias");
    CHECK_TENSOR_LOAD(final_proj_bias, "fdes_final_proj_bias");

    return true;
}

ggml_tensor* ConditionalDecoder::forward(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* mask,
    ggml_tensor* mu,
    ggml_tensor* t,
    ggml_tensor* spks,
    ggml_tensor* cond,
    ggml_tensor* r
) {
    // Time embeddings
    ggml_tensor* t_emb = sinusoidal_pos_emb(ctx, t, time_emb_dim_sinusoidal);
    t_emb = time_mlp->forward(ctx, t_emb);
    ggml_tensor* r_emb = nullptr;

    if (config.meanflow && r)
    {
        r_emb = sinusoidal_pos_emb(ctx, r, time_emb_dim_sinusoidal);
        r_emb = time_mlp->forward(ctx, r_emb);
        ggml_tensor* concat_emb = ggml_concat(ctx, t_emb, r_emb, 0);
        // Apply linear layer: t_emb = concat_emb @ time_embed_mixer_weight.T
        t_emb = ggml_mul_mat(ctx, time_embed_mixer_weight, concat_emb);
    }

    // Concatenate inputs: x, mu, spks (if provided), cond (if provided)
    x = ggml_concat(ctx, x, mu, 1);  // Concatenate along channel dimension
    
    if (spks != nullptr) {
        // Repeat spks along time dimension
        ggml_tensor* spk_shape = ggml_new_tensor_3d(ctx, spks->type, x->ne[0], spks->ne[0], spks->ne[1]);
        spks = ggml_reshape_3d(ctx, spks, 1, spks->ne[0], spks->ne[1]);
        spks = ggml_cont(ctx, spks);
        spks = ggml_repeat(ctx, spks, spk_shape);
        x = ggml_concat(ctx, x, spks, 1);
    }
    
    if (cond != nullptr) {
        x = ggml_concat(ctx, x, cond, 1);
    }
    // Down blocks
    std::vector<ggml_tensor*> hiddens;
    ggml_tensor* mask_down = mask;
    
    for (auto& down_block : down_blocks) {
        
        // Resnet
        x = down_block.resnet->forward(ctx, x, mask_down, t_emb);
        
        // x is all good till here same values in all backends
        // Transformer blocks
        x = ggml_cont(ctx, ggml_transpose(ctx, x));  // (T, C, B) -> (C, T, B)
        ggml_tensor* attn_mask = create_attention_mask(ctx, mask_down, config.causal);
        ggml_set_name(attn_mask, "attn_mask");
        for (const auto& tf_block : down_block.transformer_blocks) {
            x = tf_block->forward(ctx, x, attn_mask, t_emb);
            // return x;
            // x starts to diverge in tf_block->forward
        }
        x = ggml_cont(ctx, ggml_transpose(ctx, x));  // (B, T, C) -> (B, C, T)
        
        hiddens.push_back(x);
        
        // Downsample
        x = ggml_mul(ctx, x, mask_down);
        if (down_block.is_downsample) {
            ggml_tensor* w = (down_block.downsample_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, down_block.downsample_weight, GGML_TYPE_F32) : down_block.downsample_weight;
            x = ggml_conv_1d(ctx, w, x, 2, 1, 1);
        } else {
            if (config.causal) {
                x = apply_causal_conv1d(ctx, x, down_block.downsample_weight, down_block.downsample_bias, 3);
            } else {
                ggml_tensor* w = (down_block.downsample_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, down_block.downsample_weight, GGML_TYPE_F32) : down_block.downsample_weight;
                x = ggml_conv_1d(ctx, w, x, 1, 1, 1);
            }
        }
    }
    // return x;
    // Mid blocks
    for (auto& mid_block : mid_blocks) {
        // Resnet
        x = mid_block.resnet->forward(ctx, x, mask_down, t_emb);
        
        // Transformer blocks
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        ggml_tensor* attn_mask = create_attention_mask(ctx, mask_down, config.causal);
        
        for (const auto& tf_block : mid_block.transformer_blocks) {
            x = tf_block->forward(ctx, x, attn_mask, t_emb);
        }
        
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
    }

    // Up blocks
    ggml_tensor* mask_up = mask_down;
    for (auto& up_block : up_blocks) {
        
        ggml_tensor* skip = hiddens.back();
        hiddens.pop_back();
        
        // Concatenate with skip connection
        // Trim x to match skip size if needed
        if (x->ne[2] != skip->ne[2]) {
            x = ggml_view_3d(ctx, x, skip->ne[0], skip->ne[1], skip->ne[2], 
                           x->nb[1], x->nb[2], 0);
        }
        
        x = ggml_concat(ctx, x, skip, 1);
        
        // Resnet
        x = up_block.resnet->forward(ctx, x, mask_up, t_emb);

        // Transformer blocks
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        ggml_tensor* attn_mask = create_attention_mask(ctx, mask_up, config.causal);
        
        for (const auto& tf_block : up_block.transformer_blocks) {
            x = tf_block->forward(ctx, x, attn_mask, t_emb);
        }
        
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        
        // Upsample
        x = ggml_mul(ctx, x, mask_up);
        if (up_block.is_upsample) {
            x = up_block.upsample->forward(ctx, x);
        } else {
            if (config.causal) {
                x = apply_causal_conv1d(ctx, x, up_block.upsample_conv_weight, up_block.upsample_conv_bias, 3);
            } else {
                ggml_tensor* w = (up_block.upsample_conv_weight->type != GGML_TYPE_F32) ? ggml_cast(ctx, up_block.upsample_conv_weight, GGML_TYPE_F32) : up_block.upsample_conv_weight;
                x = ggml_conv_1d(ctx, w, x, 1, 1, 1);
            }
        }
    }
    
    // Final block
    x = final_block->forward(ctx, x, mask_up);
    
    // Final projection
    x = ggml_mul(ctx, x, mask_up);
    // Add batch to weight
    ggml_tensor* weight_unsqueezed = ggml_reshape_3d(ctx, final_proj_weight, 1, final_proj_weight->ne[0], final_proj_weight->ne[1]);
    x = apply_conv1d(ctx, x, weight_unsqueezed, final_proj_bias, 1, 0, 1, static_cast<int>(x->ne[2]));

    // Apply final mask
    x = ggml_mul(ctx, x, mask);
    
    return x;
}

ggml_tensor* ConditionalDecoder::create_attention_mask(
    ggml_context* ctx,
    ggml_tensor* mask,
    bool causal
) {
    // Convert padding mask to attention bias for flash attention
    // Input mask: 1 = valid position (attend), 0 = padded position (mask out)
    // Output: 0 for valid positions, -inf for masked positions
    // Formula: (1 - mask) * -inf
    
// Pad to multiple of 256 for flash attention compatibility FIRST
// ggml_pad fills with 0, which means "padded/masked" in our mask format
auto seq_len = static_cast<int>(mask->ne[0]);
auto padded_seq_len = static_cast<int>(GGML_PAD(seq_len, 256));
int pad_amount = padded_seq_len - seq_len;
    
    ggml_tensor* padded_mask = mask;
    if (pad_amount > 0) {
        padded_mask = ggml_pad(ctx, mask, pad_amount, 0, 0, 0);
    }
    
    // Create ones tensor with same shape as padded mask
    ggml_tensor* ones = ggml_repeat(ctx, ggml_arange(ctx, 1.0f, 2.0f, 1.0f), padded_mask);
    
    // Compute (1 - mask) * -inf
    // For valid positions (mask=1): (1-1)*-inf = 0 (attend)
    // For padded positions (mask=0): (1-0)*-inf = -inf (completely mask out)
    // Using -FLT_MAX as a portable large negative value (softmax(-inf) = 0)
    constexpr float NEG_INF = -3.402823466e+38f; // -FLT_MAX
    ggml_tensor* attn_mask = ggml_scale(ctx, ggml_sub(ctx, ones, padded_mask), NEG_INF);
    
    // TODO: Handle causal masking if needed
    (void)causal;  // Suppress unused parameter warning for now
    
    return attn_mask;
}

void ConditionalDecoder::print_model_info() const {
    printf("ConditionalDecoder Configuration:\n");
    printf("  in_channels: %d\n", config.decoder_in_channels);
    printf("  out_channels: %d\n", config.decoder_out_channels);
    printf("  causal: %s\n", config.causal ? "true" : "false");
    printf("  channels: [");
    for (size_t i = 0; i < config.channels.size(); i++) {
        printf("%d", config.channels[i]);
        if (i < config.channels.size() - 1) printf(", ");
    }
    printf("]\n");
    printf("  dropout: %f\n", config.dropout);
    printf("  attention_head_dim: %d\n", config.attention_head_dim);
    printf("  n_blocks: %d\n", config.n_blocks);
    printf("  num_mid_blocks: %d\n", config.num_mid_blocks);
    printf("  num_heads: %d\n", config.num_heads);
    printf("  act_fn: %s\n", config.act_fn.c_str());
    printf("  down_blocks: %zu\n", down_blocks.size());
    printf("  mid_blocks: %zu\n", mid_blocks.size());
    printf("  up_blocks: %zu\n", up_blocks.size());
}

void ConditionalDecoder::print_shape(const char* name, const ggml_tensor* t) const {
    if (!t) {
        printf("%s: nullptr\n", name);
        return;
    }
    
    printf("%s: shape=[", name);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        printf("%lld", t->ne[i]);
        if (i < GGML_MAX_DIMS - 1) printf(", ");
    }
    printf("]\n");
}

ggml_tensor* ConditionalDecoder::checked_get_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    if (!tensor) {
        fprintf(stderr, "Error: tensor '%s' not found\n", name);
    }
    return tensor;
}

// ============================================================================
// Flow Matching (absorbed from CausalConditionalCFM)
// ============================================================================

static ggml_tensor* apply_cosine_schedule(ggml_context* ctx, ggml_tensor* t_span) {
    float half_pi = 2.0f * static_cast<float>(asin(1.0)) / 2.0f;
    ggml_tensor* scaled = ggml_scale(ctx, t_span, half_pi);
    ggml_tensor* cos_t = ggml_cos(ctx, scaled);
    ggml_tensor* ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, t_span->ne[0]);
    ones = ggml_repeat(ctx, ggml_arange(ctx, 1, 2, 1), ones);
    return ggml_sub(ctx, ones, cos_t);
}

std::pair<ggml_tensor*, ggml_tensor*> ConditionalDecoder::forward_causal(
    ggml_context* ctx, ggml_tensor* mu, ggml_tensor* mask,
    int n_timesteps, float temperature,
    ggml_tensor* spks, ggml_tensor* cond
) {
    int mu_time_dim = static_cast<int>(mu->ne[0]);
    ggml_tensor* z = ggml_cont(ctx, ggml_view_3d(
        ctx, rand_noise, mu_time_dim, 80, 1,
        rand_noise->nb[1], rand_noise->nb[2], 0));
    if (temperature != 1.0f) {
        z = ggml_scale(ctx, z, temperature);
    }

    float step = 1.0f / n_timesteps;
    ggml_tensor* t_span = ggml_arange(ctx, 0.0f, 1.0f + 1e-5f, step);
    ggml_set_name(t_span, "t_span");

    if (!config.meanflow && strcmp(config.t_scheduler, "cosine") == 0) {
        t_span = apply_cosine_schedule(ctx, t_span);
    }

    ggml_tensor* result = nullptr;
    if (config.meanflow) {
        result = basic_euler(ctx, z, t_span, mu, mask, spks, cond);
    } else {
        result = solve_euler(ctx, z, t_span, mu, mask, spks, cond);
    }

    return std::make_pair(result, t_span);
}

ggml_tensor* ConditionalDecoder::solve_euler(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* t_span,
    ggml_tensor* mu, ggml_tensor* mask,
    ggml_tensor* spks, ggml_tensor* cond
) {
    int n_steps = static_cast<int>(t_span->ne[0]);
    ggml_tensor* t = ggml_cont(ctx, ggml_view_1d(ctx, t_span, 1, 0));
    ggml_tensor* dt = ggml_cont(ctx, ggml_view_1d(ctx, t_span, 1, t_span->nb[0]));
    dt = ggml_sub(ctx, dt, t);

    int n_feats = 80;
    int time_steps = static_cast<int>(mu->ne[0]);

    ggml_tensor* x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, time_steps, n_feats, 2);
    ggml_tensor* mask_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, time_steps, 1, 2);
    ggml_tensor* mu_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, time_steps, n_feats, 1);
    ggml_tensor* t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor* spks_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 80, 1);
    ggml_tensor* cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, time_steps, n_feats);

    for (int step_i = 1; step_i < n_steps; step_i++) {
        x_in = ggml_repeat(ctx, x, x_in);
        mask_in = ggml_repeat(ctx, mask, mask_in);
        ggml_tensor* mu_in_cat = ggml_concat(ctx, mu, mu_in, 2);
        t_in = ggml_repeat(ctx, t, t_in);
        ggml_tensor* spks_in_cat = ggml_concat(ctx, spks, spks_in, 1);
        ggml_tensor* cond_in_cat = ggml_concat(ctx, cond, cond_in, 2);

        ggml_tensor* dphi_dt = forward(ctx, x_in, mask_in, mu_in_cat, t_in, spks_in_cat, cond_in_cat, nullptr);

        ggml_tensor* dphi_dt_cond = ggml_cont(ctx, ggml_view_3d(ctx, dphi_dt, time_steps, n_feats, 1,
            dphi_dt->nb[1], dphi_dt->nb[2], 0));
        ggml_tensor* dphi_dt_uncond = ggml_cont(ctx, ggml_view_3d(ctx, dphi_dt, time_steps, n_feats, 1,
            dphi_dt->nb[1], dphi_dt->nb[2], dphi_dt->nb[2]));

        float cfg_scale = 1.0f + config.inference_cfg_rate;
        ggml_tensor* scaled_cond = ggml_scale(ctx, dphi_dt_cond, cfg_scale);
        ggml_tensor* scaled_uncond = ggml_scale(ctx, dphi_dt_uncond, config.inference_cfg_rate);
        ggml_tensor* dphi_dt_final = ggml_sub(ctx, scaled_cond, scaled_uncond);

        ggml_tensor* dx = ggml_mul(ctx, dphi_dt_final, dt);
        x = ggml_add(ctx, x, dx);

        t = ggml_add(ctx, t, dt);
        if (step_i < n_steps - 1) {
            dt = ggml_sub(ctx, ggml_cont(ctx, ggml_view_1d(ctx, t_span, 1, (step_i + 1) * t_span->nb[0])), t);
        }
    }
    return x;
}

ggml_tensor* ConditionalDecoder::basic_euler(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* time_span,
    ggml_tensor* mu, ggml_tensor* mask,
    ggml_tensor* spks, ggml_tensor* cond
) {
    auto n_steps = static_cast<int>(time_span->ne[0]);
    printf("S3 Token -> Mel Inference (basic_euler)...\n");

    for (int step = 0; step < n_steps - 1; step++) {
        ggml_tensor* t = ggml_cont(ctx, ggml_view_1d(ctx, time_span, 1, step * time_span->nb[0]));
        ggml_tensor* r = ggml_cont(ctx, ggml_view_1d(ctx, time_span, 1, (step + 1) * time_span->nb[0]));
        ggml_tensor* dt = ggml_sub(ctx, r, t);
        ggml_tensor* dxdt = forward(ctx, x, mask, mu, t, spks, cond, r);
        ggml_tensor* dx = ggml_mul(ctx, dxdt, dt);
        x = ggml_add(ctx, x, dx);
    }
    return x;
}

// ============================================================================
// BasicTransformerBlock Implementation
// ============================================================================

BasicTransformerBlock::BasicTransformerBlock(
    int dim_, int num_attention_heads_, int attention_head_dim_,
    float dropout_, const char* activation_fn_
) : dim(dim_), num_attention_heads(num_attention_heads_),
    attention_head_dim(attention_head_dim_), dropout(dropout_),
    activation_fn(activation_fn_),
    norm1_weight(nullptr), norm1_bias(nullptr),
    norm2_weight(nullptr), norm2_bias(nullptr),
    norm3_weight(nullptr), norm3_bias(nullptr),
    attn1_to_q_weight(nullptr), attn1_to_k_weight(nullptr),
    attn1_to_v_weight(nullptr), attn1_to_out_weight(nullptr),
    attn1_to_out_bias(nullptr),
    ff_net_0_proj_weight(nullptr), ff_net_0_proj_bias(nullptr),
    ff_net_2_weight(nullptr), ff_net_2_bias(nullptr) {}

ggml_tensor* BasicTransformerBlock::forward(
    ggml_context* ctx, ggml_tensor* hidden_states,
    ggml_tensor* attention_mask, ggml_tensor* timestep,
    bool use_flash_attention
) {
    ggml_tensor* norm_hidden = apply_layer_norm(ctx, hidden_states, norm1_weight, norm1_bias, 1e-5f);
    ggml_tensor* q = ggml_mul_mat(ctx, attn1_to_q_weight, norm_hidden);
    ggml_tensor* k = ggml_mul_mat(ctx, attn1_to_k_weight, norm_hidden);
    ggml_tensor* v = ggml_mul_mat(ctx, attn1_to_v_weight, norm_hidden);
    int batch = static_cast<int>(hidden_states->ne[2]);
    int time = static_cast<int>(hidden_states->ne[1]);
    int head_dim = static_cast<int>(k->ne[0]) / num_attention_heads;
    int batch_size = static_cast<int>(k->ne[2]);
    q = ggml_cont(ctx, ggml_reshape_4d(ctx, q, head_dim, num_attention_heads, time, batch));
    k = ggml_cont(ctx, ggml_reshape_4d(ctx, k, head_dim, num_attention_heads, time, batch));
    v = ggml_cont(ctx, ggml_reshape_4d(ctx, v, head_dim, num_attention_heads, time, batch));
    attention_mask = ggml_cont(ctx, attention_mask);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* attn_output = nullptr;
    if (!use_flash_attention) {
        ggml_tensor* scores = ggml_mul_mat(ctx, q, k);
        scores = ggml_scale(ctx, scores, 1.0f / sqrtf(static_cast<float>(head_dim)));
        scores = ggml_cont(ctx, ggml_transpose(ctx, scores));
        if (attention_mask) scores = ggml_add(ctx, scores, attention_mask);
        attn_output = ggml_mul_mat(ctx, ggml_soft_max(ctx, scores), ggml_cont(ctx, ggml_transpose(ctx, v)));
        attn_output = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_3d(ctx, attn_output, time, head_dim * num_attention_heads, batch)));
    } else {
        q = ggml_cont(ctx, ggml_reshape_3d(ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]));
        k = ggml_cont(ctx, ggml_reshape_3d(ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]));
        v = ggml_cont(ctx, ggml_reshape_3d(ctx, v, v->ne[0], v->ne[1], v->ne[2] * v->ne[3]));
        auto padded_kv_len = static_cast<int>(GGML_PAD(k->ne[1], 256));
        k = ggml_cast(ctx, ggml_pad(ctx, k, 0, padded_kv_len - static_cast<int>(k->ne[1]), 0, 0), GGML_TYPE_F16);
        v = ggml_cast(ctx, ggml_pad(ctx, v, 0, padded_kv_len - static_cast<int>(v->ne[1]), 0, 0), GGML_TYPE_F16);
        constexpr int KQ_MASK_PAD = 64;
        auto q_seq_padded = static_cast<int>(GGML_PAD(static_cast<int>(q->ne[1]), KQ_MASK_PAD));
        ggml_tensor* flash_mask = nullptr;
        if (attention_mask && static_cast<int>(attention_mask->ne[0]) >= padded_kv_len) {
            flash_mask = ggml_reshape_4d(ctx, ggml_view_1d(ctx, attention_mask, padded_kv_len, 0), padded_kv_len, 1, 1, 1);
            flash_mask = ggml_cast(ctx, ggml_clamp(ctx,
                ggml_repeat(ctx, flash_mask, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, padded_kv_len, q_seq_padded, 1, 1)),
                -65504.0f, 0.0f), GGML_TYPE_F16);
        }
        attn_output = ggml_flash_attn_ext(ctx, q, k, v, flash_mask, 1.0f / sqrtf(static_cast<float>(head_dim)), 0.0f, 0.0f);
        attn_output = ggml_cont(ctx, ggml_permute(ctx,
            ggml_view_4d(ctx, attn_output, attn_output->ne[0] * attn_output->ne[1] / batch_size, batch_size, attn_output->ne[2], attn_output->ne[3],
                attn_output->nb[2] / 2, attn_output->nb[2], attn_output->nb[3], 0),
            0, 2, 1, 3));
    }
    attn_output = ggml_add(ctx, ggml_mul_mat(ctx, attn1_to_out_weight, attn_output), attn1_to_out_bias);
    hidden_states = ggml_add(ctx, attn_output, hidden_states);
    norm_hidden = apply_layer_norm(ctx, hidden_states, norm3_weight, norm3_bias, 1e-5);
    ggml_tensor* ff = ggml_add(ctx, ggml_mul_mat(ctx, ff_net_0_proj_weight, norm_hidden), ff_net_0_proj_bias);
    ff = (activation_fn == "gelu") ? ggml_gelu(ctx, ff) : apply_silu(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, ff_net_2_weight, ff), ff_net_2_bias);
    return ggml_add(ctx, ff, hidden_states);
}

bool BasicTransformerBlock::load_weights(ggml_context* model_ctx, const char* prefix) {
    char name[256];
    snprintf(name, sizeof(name), "%s_norm1_weight", prefix); norm1_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(norm1_weight, name);
    snprintf(name, sizeof(name), "%s_norm1_bias", prefix); norm1_bias = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(norm1_bias, name);
    snprintf(name, sizeof(name), "%s_norm3_weight", prefix); norm3_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(norm3_weight, name);
    snprintf(name, sizeof(name), "%s_norm3_bias", prefix); norm3_bias = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(norm3_bias, name);
    snprintf(name, sizeof(name), "%s_attn1_to_q_weight", prefix); attn1_to_q_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(attn1_to_q_weight, name);
    snprintf(name, sizeof(name), "%s_attn1_to_k_weight", prefix); attn1_to_k_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(attn1_to_k_weight, name);
    snprintf(name, sizeof(name), "%s_attn1_to_v_weight", prefix); attn1_to_v_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(attn1_to_v_weight, name);
    snprintf(name, sizeof(name), "%s_attn1_to_out_0_weight", prefix); attn1_to_out_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(attn1_to_out_weight, name);
    snprintf(name, sizeof(name), "%s_attn1_to_out_0_bias", prefix); attn1_to_out_bias = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(attn1_to_out_bias, name);
    snprintf(name, sizeof(name), "%s_ff_net_0_proj_weight", prefix); ff_net_0_proj_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(ff_net_0_proj_weight, name);
    snprintf(name, sizeof(name), "%s_ff_net_0_proj_bias", prefix); ff_net_0_proj_bias = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(ff_net_0_proj_bias, name);
    snprintf(name, sizeof(name), "%s_ff_net_2_weight", prefix); ff_net_2_weight = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(ff_net_2_weight, name);
    snprintf(name, sizeof(name), "%s_ff_net_2_bias", prefix); ff_net_2_bias = ggml_get_tensor(model_ctx, name); CHECK_TENSOR_LOAD(ff_net_2_bias, name);
    return true;
}
