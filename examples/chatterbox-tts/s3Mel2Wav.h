#pragma once

#include "s3gen_ops.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <cstdint>
#include <string>

// Forward declarations
struct ggml_context;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

// ============================================================================
// Model data types (from s3gen_types.h)
// ============================================================================

struct S3GenMelsData {
    std::vector<float> data;
    std::vector<uint32_t> shape;

    bool is_valid() const {
        if (data.empty() || shape.empty()) return false;
        uint64_t expected_size = 1;
        for (uint32_t dim : shape) expected_size *= dim;
        return expected_size == data.size();
    }

    void print_info() const {
        printf("S3GenMelsData - Shape: ");
        for (size_t i = 0; i < shape.size(); i++) {
            printf("%u", shape[i]);
            if (i < shape.size() - 1) printf(" x ");
        }
        printf(", Data size: %zu elements\n", data.size());
    }
};

struct layer {
    ggml_tensor* weight_v;
    ggml_tensor* weight_g;
    ggml_tensor* bias;
};

struct f0_predictor {
    std::vector<layer> conv_layers;
    ggml_tensor* classifier_weight;
    ggml_tensor* classifier_bias;
};

struct source_module {
    ggml_tensor* linear_weight;
    ggml_tensor* linear_bias;
    ggml_tensor* noise_tensor;
};

struct resblock {
    std::vector<layer> layer_group_1;
    std::vector<layer> layer_group_2;
    std::vector<ggml_tensor*> activations_alpha_1;
    std::vector<ggml_tensor*> activation_alpha_2;
};

struct source_resblock {
    std::vector<layer> layer_group_1;
    std::vector<layer> layer_group_2;
    std::vector<ggml_tensor*> activations_alpha_1;
    std::vector<ggml_tensor*> activation_alpha_2;
};

struct decode_module {
    layer pre_conv_layer;
    layer post_conv_layer;
    std::vector<resblock> resblocks;
    std::vector<source_resblock> source_resblocks;
    std::vector<layer> up_layers;
    std::vector<layer> down_layers;
};

struct istft_module {
    ggml_tensor* hann_window;
    ggml_tensor* cos_matrix;
    ggml_tensor* sin_matrix;
};

struct S3GenModelData {
    std::string model_path;
    bool loaded = false;
    uint32_t num_tensors = 0;
    std::vector<std::string> tensor_names;

    ggml_context* ctx = nullptr;
    ggml_backend* backend = nullptr;
    ggml_backend_buffer* buffer = nullptr;

    f0_predictor f0pred;
    source_module source;
    decode_module decode;
    istft_module istft;

    void print_info() const;
    ggml_tensor* get_tensor(const char* name);
    void free();
};

struct S3GenInputData {
    S3GenMelsData mels;
    S3GenModelData model;
    bool is_complete() const { return mels.is_valid() && model.loaded; }
    void print_summary() const {
        printf("=== S3Gen Input Data Summary ===\n");
        mels.print_info();
        model.print_info();
        printf("Complete: %s\n", is_complete() ? "Yes" : "No");
    }
};

// ============================================================================
// Data loading (from s3gen_data.h)
// ============================================================================

class S3GenDataLoader {
public:
    static S3GenMelsData load_mels_from_npy(const std::string& filename);
    static S3GenModelData load_model_metadata(const std::string& filename);
    static S3GenModelData load_model_with_backend(const std::string& filename);
    static void organize_f0_predictor(S3GenModelData& model_data);
    static void organize_source_module(S3GenModelData& model_data);
    static void organize_decode_module(S3GenModelData& model_data);
    static void create_gaussian_noise_tensor(S3GenModelData& model_data);
    static void fill_gaussian_noise_tensor(S3GenModelData& model_data);
    static void create_istft_context(S3GenModelData& model_data);
    static void fill_istft_context(S3GenModelData& model_data);
    static S3GenInputData load_s3gen_inputs(const std::string& model_path, const std::string& mels_path, bool full_load = false);

private:
    static void read_bytes(std::ifstream& file, char* buffer, size_t size);
    static uint32_t read_uint32(std::ifstream& file);
    static uint16_t read_uint16(std::ifstream& file);
    static void validate_npy_header(std::ifstream& file, std::vector<uint32_t>& shape);
    static ggml_backend_t init_backend();
};

// ============================================================================
// HiFT vocoder inference (from s3gen_hift.h)
// ============================================================================

struct HiFTInferenceGraphResult {
    ggml_cgraph* graph;
    ggml_tensor* output;
    ggml_tensor* test;
};

ggml_tensor* decode(ggml_context* ctx, ggml_tensor* mels, const S3GenModelData& model);

HiFTInferenceGraphResult build_hift_inference_graph(
    ggml_context* ctx, const S3GenModelData& model, ggml_tensor* mel_input);

ggml_tensor* apply_resblock(ggml_context* ctx, ggml_tensor* x, const resblock& block, int kernel_size);
ggml_tensor* apply_source_resblock(ggml_context* ctx, ggml_tensor* x, const source_resblock& block, int kernel_size);
ggml_tensor* istft(ggml_context* ctx, ggml_tensor* magnitude, ggml_tensor* phase);
