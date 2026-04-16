#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

#include "ggml.h"
#include "ggml-backend.h"
#include "s3Token2Mel.h"
#include "s3Mel2Wav.h"

struct S3Token2Wav {
    // Per-call generation parameters (dynamic, user-managed buffers)
    struct GenerateParams {
        const int64_t* prompt_tokens = nullptr;
        const int32_t* prompt_tokens_shape = nullptr;
        size_t prompt_tokens_n_dims = 0;

        const float* prompt_feat = nullptr;
        const int32_t* prompt_feat_shape = nullptr;
        size_t prompt_feat_n_dims = 0;

        const float* speaker_emb = nullptr;
        const int32_t* speaker_emb_shape = nullptr;
        size_t speaker_emb_n_dims = 0;

        float cfg_weight = 3.0f;
    };

    explicit S3Token2Wav(ggml_backend_t shared_backend = nullptr);
    ~S3Token2Wav();

    bool init_backend();
    bool load_model(const std::string& model_path, bool is_meanflow);
    void free_model();

    bool generate(
        const std::vector<int>& speech_tokens,
        const GenerateParams& gen_params,
        float** output_audio,
        size_t* audio_size);

    // Low-level graph building (used by s3T2W_main for benchmarking)
    bool build_graph(ggml_context* ctx, ggml_tensor* token,
        ggml_tensor* prompt_token, ggml_tensor* prompt_feat, ggml_tensor* embedding,
        ggml_tensor* attention_mask, bool meanflow);

    bool compute_graph();
    void free_graph();

    ggml_backend_t backend = nullptr;
    bool owns_backend = true;
    ggml_context* model_ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_backend_buffer* buffer = nullptr;
    ggml_gallocr_t allocr = nullptr;

    std::unique_ptr<S3Token2Mel> flow;
    std::unique_ptr<S3GenModelData> mel2wav;

    ggml_context* graph_ctx = nullptr;
};
