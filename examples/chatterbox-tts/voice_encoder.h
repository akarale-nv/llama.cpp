#pragma once

// Chatterbox reference-audio encoder.
//
// Given a 16 kHz mono reference WAV, this module runs two encoders in a
// single ggml compute graph and returns two named outputs:
//
//   1) "cond_spkr" : f32, shape [1024]
//        VE (3-layer LSTM speaker encoder) + linear projection through
//        cond_enc.spkr_enc. This is row 0 of T3's cond_emb.
//
//   2) "prompt_tokens" : i32, shape [T']
//        S3TokenizerV2 (conv stem + 6 residual FSMN-attn blocks + FSQ).
//        These are the ~25/sec speech token IDs the reference clip decodes
//        into. The T3 caller looks them up in speech_emb to build cond_emb
//        rows 1..N-1, and S3Gen consumes them directly.
//
// All weights and the two mel filterbanks live in a single voice_encoder.gguf
// emitted by convert_chatterbox_to_gguf.py's `ve` subcommand.

#include "staged_io.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;

// ---- VE (LSTM speaker encoder) hparams -----------------------------------
struct VoiceEncoderConfig {
    // Mel DSP
    int   num_mels           = 40;
    int   sample_rate        = 16000;
    int   n_fft              = 400;
    int   hop_size           = 160;
    int   win_size           = 400;
    int   fmin               = 0;
    int   fmax               = 8000;
    float mel_power          = 2.0f;
    float stft_magnitude_min = 1e-4f;
    float preemphasis        = 0.0f;
    bool  normalized_mels    = false;
    std::string mel_type     = "amp";

    // LSTM speaker encoder
    int  hidden_size        = 256;
    int  num_layers         = 3;
    int  speaker_embed_size = 256;
    int  partial_frames     = 160;
    bool final_relu         = true;

    // cond_enc.spkr_enc
    int cond_enc_n_channels = 1024;
};

// ---- S3TokenizerV2 hparams ----------------------------------------------
struct S3TokConfig {
    // Mel DSP (log-mel, 128 bins, periodic hann)
    int   n_mels  = 128;
    int   n_fft   = 400;
    int   hop     = 160;
    int   sr      = 16000;

    // Encoder
    int   n_state       = 1280;
    int   n_head        = 20;
    int   n_layer       = 6;
    int   fsmn_kernel   = 31;
    int   conv1_stride  = 2;  // Overall stride is conv1_stride * 2

    // FSQ head
    int   fsq_dim   = 8;
    int   fsq_level = 3;      // per-channel codebook levels {0, 1, 2}
    float fsq_scale = 0.9990000128746033f;

    // Token rate assumptions (fixed for v2 "25hz" model).
    int   token_rate      = 25;    // tokens/sec at 16 kHz
    int   samples_per_tok = 640;   // sr / token_rate; wav is padded to a multiple of this
};

struct VoiceEncoder {
    explicit VoiceEncoder(ggml_backend_t shared_backend = nullptr);
    ~VoiceEncoder();

    bool init_backend();

    // Loads BOTH VE and S3Tok weights + both mel filterbanks from a single
    // voice_encoder.gguf produced by the `ve` conversion subcommand.
    bool load_model(const std::string & gguf_path);

    void free_model();

    // Runs VE + cond_enc + S3Tok in one ggml graph.
    // Populates `outputs` with:
    //   outputs["cond_spkr"]      : f32 [cond_n_channels]
    //   outputs["prompt_tokens"]  : i32 [T']
    // Fails on any WAV I/O error, sample-rate mismatch, or compute failure.
    bool encode_reference(const std::string & wav_path,
                          std::map<std::string, staged_io> & outputs);

    // ---- Loaded state ----
    VoiceEncoderConfig ve_cfg;
    S3TokConfig        s3_cfg;

    ggml_backend_t        backend      = nullptr;
    bool                  owns_backend = true;
    ggml_context *        model_ctx    = nullptr;
    ggml_backend_buffer_t buffer       = nullptr;

    // VE LSTM per-layer weights
    struct LstmLayer {
        ggml_tensor * w_ih = nullptr; // (in, 4H)
        ggml_tensor * b_ih = nullptr; // (4H)
        ggml_tensor * w_hh = nullptr; // (H, 4H)
        ggml_tensor * b_hh = nullptr; // (4H)
    };
    std::vector<LstmLayer> lstm;

    ggml_tensor * proj_w      = nullptr;  // VE final projection
    ggml_tensor * proj_b      = nullptr;
    ggml_tensor * ve_mels     = nullptr;  // Slaney mel filterbank (n_freqs, n_mels)
    ggml_tensor * cond_spkr_w = nullptr;  // cond_enc.spkr_enc Linear
    ggml_tensor * cond_spkr_b = nullptr;

    // S3Tok conv stem + per-block state + FSQ head
    struct S3TokBlock {
        ggml_tensor * ln1_w      = nullptr; ggml_tensor * ln1_b      = nullptr;
        ggml_tensor * attn_q_w   = nullptr; ggml_tensor * attn_q_b   = nullptr;
        ggml_tensor * attn_k_w   = nullptr; // K has no bias
        ggml_tensor * attn_v_w   = nullptr; ggml_tensor * attn_v_b   = nullptr;
        ggml_tensor * attn_out_w = nullptr; ggml_tensor * attn_out_b = nullptr;
        ggml_tensor * fsmn_conv  = nullptr; // depthwise conv, kernel=31
        ggml_tensor * ln2_w      = nullptr; ggml_tensor * ln2_b      = nullptr;
        ggml_tensor * ffn_up_w   = nullptr; ggml_tensor * ffn_up_b   = nullptr;
        ggml_tensor * ffn_down_w = nullptr; ggml_tensor * ffn_down_b = nullptr;
    };
    std::vector<S3TokBlock> s3_blocks;
    ggml_tensor * s3_conv1_w = nullptr; ggml_tensor * s3_conv1_b = nullptr;
    ggml_tensor * s3_conv2_w = nullptr; ggml_tensor * s3_conv2_b = nullptr;
    ggml_tensor * s3_mels    = nullptr; // 128-bin librosa mel filterbank
    ggml_tensor * s3_fsq_w   = nullptr; ggml_tensor * s3_fsq_b   = nullptr;
};
