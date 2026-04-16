#pragma once

#include <string>
#include <vector>

struct S3GenConfig {
    // --- Global ---
    bool meanflow = false;

    // --- Flow (S3Token2Mel) ---
    int input_size = 512;              // Token embedding size
    int output_size = 80;              // Mel features (n_mels)
    int spk_embed_dim = 192;           // Speaker embedding dimension (flow projection)
    int token_mel_ratio = 2;           // Tokens-to-mel-frame ratio
    int pre_lookahead_len = 3;         // Lookahead trimming length

    // --- CFM / Flow Matching ---
    int n_timesteps = 5;               // ODE solver steps (meanflow: 2)
    float inference_cfg_rate = 0.7f;   // Classifier-free guidance rate
    const char* t_scheduler = "cosine";

    // --- Decoder / UNet ---
    int decoder_in_channels = 320;     // UNet input channels (mel+mu+spks+cond)
    int decoder_out_channels = 80;     // UNet output channels (mel)
    bool causal = true;                // Causal attention masking
    std::vector<int> channels = {256}; // Channel progression in down/up blocks
    float dropout = 0.0f;
    int attention_head_dim = 64;
    int n_blocks = 4;                  // Transformer blocks per stage
    int num_mid_blocks = 12;
    int num_heads = 8;
    std::string act_fn = "gelu";

    // --- Encoder / UpsampleConformer ---
    int encoder_output_size = 512;
    int encoder_attention_heads = 8;
    int encoder_linear_units = 2048;
    int encoder_num_blocks = 6;
    int encoder_num_up_blocks = 4;
    int encoder_upsample_stride = 2;
    bool encoder_normalize_before = true;
    bool encoder_macaron_style = false;
    bool encoder_use_cnn_module = false;

    static S3GenConfig default_config() {
        return S3GenConfig();
    }

    static S3GenConfig meanflow_config() {
        S3GenConfig cfg;
        cfg.meanflow = true;
        cfg.n_timesteps = 2;
        return cfg;
    }
};
