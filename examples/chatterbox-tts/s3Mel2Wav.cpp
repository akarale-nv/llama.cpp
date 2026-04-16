#include "s3Mel2Wav.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Implementation of S3GenModelData methods
void S3GenModelData::print_info() const {
    printf("S3GenModelData - Path: %s\n", model_path.c_str());
    printf("  Loaded: %s\n", loaded ? "true" : "false");
    printf("  Num tensors: %u\n", num_tensors);
    printf("  Backend: %s\n", backend ? "Initialized" : "Not initialized");
}

ggml_tensor* S3GenModelData::get_tensor(const char* name) {
    if (!ctx) return nullptr;
    return ggml_get_tensor(ctx, name);
}

void S3GenModelData::free() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (backend) {
        ggml_backend_free(backend);
        backend = nullptr;
    }
}

// Load mel-spectrogram data from NumPy .npy file
S3GenMelsData S3GenDataLoader::load_mels_from_npy(const std::string& filename) {
    S3GenMelsData mels_data;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open mels file: " + filename);
    }
    
    try {
        // Validate and parse NPY header
        validate_npy_header(file, mels_data.shape);
        
        // Calculate total number of elements
        uint64_t total_elements = 1;
        for (uint32_t dim : mels_data.shape) {
            total_elements *= dim;
        }
        
        // Read the data
        mels_data.data.resize(total_elements);
        file.read(reinterpret_cast<char*>(mels_data.data.data()), total_elements * sizeof(float));
        
        if (!file.good()) {
            throw std::runtime_error("Failed to read mels data from: " + filename);
        }
        
        printf("Loaded mels data: %s\n", filename.c_str());
        mels_data.print_info();
        
    } catch (const std::exception& e) {
        file.close();
        throw std::runtime_error("Error loading mels file '" + filename + "': " + e.what());
    }
    
    file.close();
    return mels_data;
}

// Load model metadata from GGUF file  
S3GenModelData S3GenDataLoader::load_model_metadata(const std::string& filename) {
    S3GenModelData model_data;
    model_data.model_path = filename;
    
    // Initialize GGUF context to read metadata
    struct ggml_context *tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,  // Don't allocate tensors, just read metadata
        /*.ctx        =*/ &tmp_ctx,
    };
    
    gguf_context *gguf_ctx = gguf_init_from_file(filename.c_str(), gguf_params);
    if (!gguf_ctx) {
        throw std::runtime_error("Failed to open GGUF file: " + filename);
    }
    
    try {
        // Read model metadata
        model_data.num_tensors = static_cast<uint32_t>(gguf_get_n_tensors(gguf_ctx));
        
        // Get tensor names
        for (uint32_t i = 0; i < model_data.num_tensors; i++) {
            const char* name = gguf_get_tensor_name(gguf_ctx, i);
            if (name) {
                model_data.tensor_names.push_back(std::string(name));
            }
        }
        
        model_data.loaded = true;
        printf("Loaded model metadata: %s\n", filename.c_str());
        model_data.print_info();
        
    } catch (const std::exception& e) {
        gguf_free(gguf_ctx);
        ggml_free(tmp_ctx);
        throw std::runtime_error("Error reading model metadata from '" + filename + "': " + e.what());
    }
    
    // Cleanup
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    
    return model_data;
}

// Initialize backend (same as T3Model)
ggml_backend_t S3GenDataLoader::init_backend() {
    ggml_backend_t backend = nullptr;
    
    fprintf(stderr, "S3GenDataLoader::init_backend() called\n");
    
#ifdef GGML_USE_CUDA
    // Disable CUDA graphs to allow UNARY operations (sqrt, abs, elu)
#ifdef _WIN32
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
    fprintf(stderr, "S3GenDataLoader: GGML_USE_CUDA is defined, trying CUDA backend...\n");
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "S3GenDataLoader: ggml_backend_cuda_init() failed\n");
    } else {
        fprintf(stderr, "S3GenDataLoader: CUDA backend initialized successfully\n");
    }
#else
    fprintf(stderr, "S3GenDataLoader: GGML_USE_CUDA is NOT defined\n");
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "S3GenDataLoader: GGML_USE_METAL is defined, trying Metal backend...\n");
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "S3GenDataLoader: ggml_backend_metal_init() failed\n");
    } else {
        fprintf(stderr, "S3GenDataLoader: Metal backend initialized successfully\n");
    }
#else
    fprintf(stderr, "S3GenDataLoader: GGML_USE_METAL is NOT defined\n");
#endif

#ifdef GGML_USE_VULKAN
    fprintf(stderr, "S3GenDataLoader: GGML_USE_VULKAN is defined\n");
    if (!backend) {
        fprintf(stderr, "S3GenDataLoader: Attempting to initialize Vulkan backend...\n");
        backend = ggml_backend_vk_init(0); // init device 0
        if (!backend) {
            fprintf(stderr, "S3GenDataLoader: ggml_backend_vk_init() FAILED\n");
        } else {
            fprintf(stderr, "S3GenDataLoader: Vulkan backend initialized SUCCESSFULLY\n");
        }
    } else {
        fprintf(stderr, "S3GenDataLoader: Skipping Vulkan (backend already initialized)\n");
    }
#else
    fprintf(stderr, "S3GenDataLoader: GGML_USE_VULKAN is NOT defined\n");
#endif

    // Fallback to CPU backend if no GPU backend available
    if (!backend) {
        fprintf(stderr, "S3GenDataLoader: Falling back to CPU backend\n");
        backend = ggml_backend_cpu_init();
    }
    
    fprintf(stderr, "S3GenDataLoader::init_backend() returning backend=%p\n", (void*)backend);
    return backend;
}

// Helper function to organize F0 predictor tensors
void S3GenDataLoader::organize_f0_predictor(S3GenModelData& model_data) {
    // Load 5 convolutional layers (indices: 0, 2, 4, 6, 8)
    const int conv_indices[] = {0, 2, 4, 6, 8};
    model_data.f0pred.conv_layers.resize(5);
    
    for (int i = 0; i < 5; i++) {
        int layer_idx = conv_indices[i];
        char tensor_name[128];
        
        // Load weight_v (original1)
        snprintf(tensor_name, sizeof(tensor_name), "m2wf0p_condnet_%d_parametrizations_weight_original1", layer_idx);
        model_data.f0pred.conv_layers[i].weight_v = model_data.get_tensor(tensor_name);
        if (!model_data.f0pred.conv_layers[i].weight_v) {
            fprintf(stderr, "Error: Could not find tensor: %s\n", tensor_name);
        }
        
        // Load weight_g (original0)
        snprintf(tensor_name, sizeof(tensor_name), "m2wf0p_condnet_%d_parametrizations_weight_original0", layer_idx);
        model_data.f0pred.conv_layers[i].weight_g = model_data.get_tensor(tensor_name);
        if (!model_data.f0pred.conv_layers[i].weight_g) {
            fprintf(stderr, "Error: Could not find tensor: %s\n", tensor_name);
        }
        
        // Load bias
        snprintf(tensor_name, sizeof(tensor_name), "m2wf0p_condnet_%d_bias", layer_idx);
        model_data.f0pred.conv_layers[i].bias = model_data.get_tensor(tensor_name);
        if (!model_data.f0pred.conv_layers[i].bias) {
            fprintf(stderr, "Error: Could not find tensor: %s\n", tensor_name);
        }
    }
    
    // Load classifier tensors
    model_data.f0pred.classifier_weight = model_data.get_tensor("m2wf0p_classifier_weight");
    model_data.f0pred.classifier_bias = model_data.get_tensor("m2wf0p_classifier_bias");
    
    if (!model_data.f0pred.classifier_weight) {
        fprintf(stderr, "Error: Could not find classifier weight tensor\n");
    }
    if (!model_data.f0pred.classifier_bias) {
        fprintf(stderr, "Error: Could not find classifier bias tensor\n");
    }
}

// Organize source module weights from flat GGUF structure
void S3GenDataLoader::organize_source_module(S3GenModelData& model_data) {
    // Load linear layer weights for merging harmonics
    // m_source.l_linear: Linear(harmonic_num+1, 1) where harmonic_num=8
    model_data.source.linear_weight = model_data.get_tensor("m2ws_l_linear_weight");
    model_data.source.linear_bias = model_data.get_tensor("m2ws_l_linear_bias");
    
    if (!model_data.source.linear_weight) {
        fprintf(stderr, "Error: Could not find source module linear weight tensor\n");
    }
    if (!model_data.source.linear_bias) {
        fprintf(stderr, "Error: Could not find source module linear bias tensor\n");
    }
    
    // Note: Gaussian noise tensor is created earlier in the loading process
}

// Organize decode module tensors
void S3GenDataLoader::organize_decode_module(S3GenModelData& model_data) {
    // Load pre conv layer
    model_data.decode.pre_conv_layer.weight_v = model_data.get_tensor("m2w_conv_pre_parametrizations_weight_original1");
    if (!model_data.decode.pre_conv_layer.weight_v) {
        fprintf(stderr, "Error: Could not find decode module pre conv layer tensor\n");
    }
    model_data.decode.pre_conv_layer.weight_g = model_data.get_tensor("m2w_conv_pre_parametrizations_weight_original0");
    if (!model_data.decode.pre_conv_layer.weight_g) {
        fprintf(stderr, "Error: Could not find decode module pre conv layer weight g tensor\n");
    }
    model_data.decode.pre_conv_layer.bias = model_data.get_tensor("m2w_conv_pre_bias");
    if (!model_data.decode.pre_conv_layer.bias) {
        fprintf(stderr, "Error: Could not find decode module pre conv layer bias tensor\n");
    }
    
    // Load post conv layer
    model_data.decode.post_conv_layer.weight_v = model_data.get_tensor("m2w_conv_post_parametrizations_weight_original1");
    if (!model_data.decode.post_conv_layer.weight_v) {
        fprintf(stderr, "Error: Could not find decode module post conv layer weight v tensor\n");
    }
    model_data.decode.post_conv_layer.weight_g = model_data.get_tensor("m2w_conv_post_parametrizations_weight_original0");
    if (!model_data.decode.post_conv_layer.weight_g) {
        fprintf(stderr, "Error: Could not find decode module post conv layer weight g tensor\n");
    }
    model_data.decode.post_conv_layer.bias = model_data.get_tensor("m2w_conv_post_bias");
    if (!model_data.decode.post_conv_layer.bias) {
        fprintf(stderr, "Error: Could not find decode module post conv layer bias tensor\n");
    }

    // Load 9 regular resblocks
    model_data.decode.resblocks.resize(9);
    for (int i = 0; i < 9; i++) {
        char tensor_name[128];
        
        // Initialize layer groups with 3 layers each
        model_data.decode.resblocks[i].layer_group_1.resize(3);
        model_data.decode.resblocks[i].layer_group_2.resize(3);
        model_data.decode.resblocks[i].activations_alpha_1.resize(3);
        model_data.decode.resblocks[i].activation_alpha_2.resize(3);
        
        // Load 3 conv layers for each group
        for (int j = 0; j < 3; j++) {
            // Load convs1 (layer_group_1)
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_convs1_%d_parametrizations_weight_original1", i, j);
            model_data.decode.resblocks[i].layer_group_1[j].weight_v = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_convs1_%d_parametrizations_weight_original0", i, j);
            model_data.decode.resblocks[i].layer_group_1[j].weight_g = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_convs1_%d_bias", i, j);
            model_data.decode.resblocks[i].layer_group_1[j].bias = model_data.get_tensor(tensor_name);
            
            // Load convs2 (layer_group_2)
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_convs2_%d_parametrizations_weight_original1", i, j);
            model_data.decode.resblocks[i].layer_group_2[j].weight_v = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_convs2_%d_parametrizations_weight_original0", i, j);
            model_data.decode.resblocks[i].layer_group_2[j].weight_g = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_convs2_%d_bias", i, j);
            model_data.decode.resblocks[i].layer_group_2[j].bias = model_data.get_tensor(tensor_name);
            
            // Load activation alphas
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_activations1_%d_alpha", i, j);
            model_data.decode.resblocks[i].activations_alpha_1[j] = model_data.get_tensor(tensor_name);
            if (!model_data.decode.resblocks[i].activations_alpha_1[j]) {
                fprintf(stderr, "Error: Could not find resblock activation alpha 1 tensor\n");
            }
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_resblks_%d_activations2_%d_alpha", i, j);
            model_data.decode.resblocks[i].activation_alpha_2[j] = model_data.get_tensor(tensor_name);
            if (!model_data.decode.resblocks[i].activation_alpha_2[j]) {
                fprintf(stderr, "Error: Could not find resblock activation alpha 2 tensor\n");
            }
        }
    }
    
    // Load 3 source resblocks
    model_data.decode.source_resblocks.resize(3);
    for (int i = 0; i < 3; i++) {
        char tensor_name[128];
        
        // Initialize layer groups with 3 layers each
        model_data.decode.source_resblocks[i].layer_group_1.resize(3);
        model_data.decode.source_resblocks[i].layer_group_2.resize(3);
        model_data.decode.source_resblocks[i].activations_alpha_1.resize(3);
        model_data.decode.source_resblocks[i].activation_alpha_2.resize(3);
        
        // Load 3 conv layers for each group
        for (int j = 0; j < 3; j++) {
            // Load convs1 (layer_group_1)
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_convs1_%d_parametrizations_weight_original1", i, j);
            model_data.decode.source_resblocks[i].layer_group_1[j].weight_v = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_convs1_%d_parametrizations_weight_original0", i, j);
            model_data.decode.source_resblocks[i].layer_group_1[j].weight_g = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_convs1_%d_bias", i, j);
            model_data.decode.source_resblocks[i].layer_group_1[j].bias = model_data.get_tensor(tensor_name);
            
            // Load convs2 (layer_group_2)
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_convs2_%d_parametrizations_weight_original1", i, j);
            model_data.decode.source_resblocks[i].layer_group_2[j].weight_v = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_convs2_%d_parametrizations_weight_original0", i, j);
            model_data.decode.source_resblocks[i].layer_group_2[j].weight_g = model_data.get_tensor(tensor_name);
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_convs2_%d_bias", i, j);
            model_data.decode.source_resblocks[i].layer_group_2[j].bias = model_data.get_tensor(tensor_name);
            
            // Load activation alphas
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_activations1_%d_alpha", i, j);
            model_data.decode.source_resblocks[i].activations_alpha_1[j] = model_data.get_tensor(tensor_name);
            if (!model_data.decode.source_resblocks[i].activations_alpha_1[j]) {
                fprintf(stderr, "Error: Could not find source module resblock activation alpha 1 tensor\n");
            }
            
            snprintf(tensor_name, sizeof(tensor_name), "m2w_source_resblks_%d_activations2_%d_alpha", i, j);
            model_data.decode.source_resblocks[i].activation_alpha_2[j] = model_data.get_tensor(tensor_name);
            if (!model_data.decode.source_resblocks[i].activation_alpha_2[j]) {
                fprintf(stderr, "Error: Could not find source module resblock activation alpha 2 tensor\n");
            }
        }
    }
    
    // Load 3 up layers (upsampling transposed convolutions)
    model_data.decode.up_layers.resize(3);
    for (int i = 0; i < 3; i++) {
        char tensor_name[128];
        
        snprintf(tensor_name, sizeof(tensor_name), "m2wsups_%d_parametrizations_weight_original1", i);
        model_data.decode.up_layers[i].weight_v = model_data.get_tensor(tensor_name);
        
        snprintf(tensor_name, sizeof(tensor_name), "m2wsups_%d_parametrizations_weight_original0", i);
        model_data.decode.up_layers[i].weight_g = model_data.get_tensor(tensor_name);
        
        snprintf(tensor_name, sizeof(tensor_name), "m2wsups_%d_bias", i);
        model_data.decode.up_layers[i].bias = model_data.get_tensor(tensor_name);
    }
    
    // Load 3 down layers (source downsampling convolutions)
    model_data.decode.down_layers.resize(3);
    for (int i = 0; i < 3; i++) {
        char tensor_name[128];
        
        snprintf(tensor_name, sizeof(tensor_name), "m2wsdown_%d_weight", i);
        model_data.decode.down_layers[i].weight_v = model_data.get_tensor(tensor_name);
        
        snprintf(tensor_name, sizeof(tensor_name), "m2wsdown_%d_bias", i);
        model_data.decode.down_layers[i].bias = model_data.get_tensor(tensor_name);
        
        // Down layers don't have weight_g (no weight norm parametrization)
        model_data.decode.down_layers[i].weight_g = nullptr;
    }
    
    printf("Organized decode module: 9 resblocks, 3 source resblocks, 3 up layers, 3 down layers\n");
}

// Create Gaussian noise tensor for source module
void S3GenDataLoader::create_gaussian_noise_tensor(S3GenModelData& model_data) {
    // Standard S3Gen parameters (should match s3gen_hift.cpp)
    // TODO: Make length parameter dynamic based on the input length
    const int length = 44160;         // Output length after F0 upsampling (86 * 480)
    const int harmonic_num = 8;       // Number of harmonics
    const int total_harmonics = harmonic_num + 1;  // Including fundamental
    
    // Create the noise tensor in the model context [length, harmonics]
    model_data.source.noise_tensor = ggml_new_tensor_2d(model_data.ctx, GGML_TYPE_F32, length, total_harmonics);
    
    if (!model_data.source.noise_tensor) {
        fprintf(stderr, "Error: Failed to create Gaussian noise tensor\n");
        return;
    }
    
    // Set tensor name for identification
    ggml_set_name(model_data.source.noise_tensor, "source_gaussian_noise");
    
    printf("Created Gaussian noise tensor [%d, %d] for source module\n", length, total_harmonics);
}

void S3GenDataLoader::create_istft_context(S3GenModelData& model_data) {
    const int n_fft = 16;
    model_data.istft.cos_matrix = ggml_new_tensor_2d(model_data.ctx, GGML_TYPE_F32, n_fft, n_fft);
    model_data.istft.sin_matrix = ggml_new_tensor_2d(model_data.ctx, GGML_TYPE_F32, n_fft, n_fft);
    model_data.istft.hann_window = ggml_new_tensor_1d(model_data.ctx, GGML_TYPE_F32, n_fft);
    ggml_set_name(model_data.istft.cos_matrix, "istft_cos_matrix");
    ggml_set_name(model_data.istft.sin_matrix, "istft_sin_matrix");
    ggml_set_name(model_data.istft.hann_window, "istft_hann_window");
    printf("Created ISTFT context\n");
}

// Fill the noise tensor with Gaussian random values (called after backend allocation)
void S3GenDataLoader::fill_gaussian_noise_tensor(S3GenModelData& model_data) {
    if (!model_data.source.noise_tensor) {
        fprintf(stderr, "Error: Noise tensor not created\n");
        return;
    }
    
    // Generate Gaussian random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);  // mean=0, std=1
    
    size_t n_elements = ggml_nelements(model_data.source.noise_tensor);
    std::vector<float> noise_data(n_elements);
    
    // Fill with Gaussian random values
    for (size_t i = 0; i < n_elements; i++) {
        noise_data[i] = dist(gen);
    }
    
    // Copy data to backend tensor
    ggml_backend_tensor_set(model_data.source.noise_tensor, noise_data.data(), 0, noise_data.size() * sizeof(float));
    
    printf("Filled noise tensor with %zu Gaussian random values\n", n_elements);
}

void S3GenDataLoader::fill_istft_context(S3GenModelData& model_data) {
    const int n_fft = 16;
    size_t n_elements = ggml_nelements(model_data.istft.cos_matrix); 
    std::vector<float> cos_matrix_data(n_elements);
    std::vector<float> sin_matrix_data(n_elements);
    for (int k = 0; k < n_fft; k++) {
        for (int n = 0; n < n_fft; n++) {
            float angle = 2.0f * static_cast<float>(M_PI) * k * n / n_fft;
            cos_matrix_data[k * n_fft + n] = cosf(angle);
            sin_matrix_data[k * n_fft + n] = sinf(angle);
        }
    }
    ggml_backend_tensor_set(model_data.istft.cos_matrix, cos_matrix_data.data(), 0, cos_matrix_data.size() * sizeof(float));
    ggml_backend_tensor_set(model_data.istft.sin_matrix, sin_matrix_data.data(), 0, sin_matrix_data.size() * sizeof(float));

    std::vector<float> hann_window_data = {0.0000f, 0.0381f, 0.1464f, 0.3087f, 0.5000f, 0.6913f, 0.8536f, 0.9619f, 1.0000f,
        0.9619f, 0.8536f, 0.6913f, 0.5000f, 0.3087f, 0.1464f, 0.0381f};
   
    ggml_backend_tensor_set(model_data.istft.hann_window, hann_window_data.data(), 0, hann_window_data.size() * sizeof(float));
}

// Load full model with backend (for inference)
S3GenModelData S3GenDataLoader::load_model_with_backend(const std::string& filename) {
    S3GenModelData model_data;
    model_data.model_path = filename;
    
    // Initialize backend first
    model_data.backend = init_backend();
    if (!model_data.backend) {
        throw std::runtime_error("Failed to initialize backend");
    }
    
    // Step 1: Load GGUF into temporary context with allocation
    struct ggml_context* tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,  // Allocate tensors in temp context
        /*.ctx        =*/ &tmp_ctx,
    };
    
    gguf_context* gguf_ctx = gguf_init_from_file(filename.c_str(), gguf_params);
    if (!gguf_ctx) {
        throw std::runtime_error("Failed to open GGUF file: " + filename);
    }
    
    try {
        int num_tensors = static_cast<int>(gguf_get_n_tensors(gguf_ctx));
        printf("S3GenDataLoader: Loading %d tensors with backend...\n", num_tensors);
        
        // Step 2: Create new context for backend (no allocation yet)
        // Reserve extra memory for additional tensors (noise tensor + inference intermediates)
        size_t base_memory = ggml_tensor_overhead() * num_tensors;
        size_t extra_memory = ggml_tensor_overhead() * 200;  // Extra space for ~200 intermediate tensors
        struct ggml_init_params params {
            /*.mem_size   =*/ base_memory + extra_memory,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,  // Don't allocate, let backend do it
        };
        model_data.ctx = ggml_init(params);
        
        // Step 3: Duplicate tensor structures from temp to backend context
        for (int i = 0; i < num_tensors; i++) {
            const char* name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor* src = ggml_get_tensor(tmp_ctx, name);
            struct ggml_tensor* dst = ggml_dup_tensor(model_data.ctx, src);
            ggml_set_name(dst, name);
            
            model_data.tensor_names.push_back(std::string(name));
        }
        
        // Step 4: Create additional tensors before backend allocation
        create_gaussian_noise_tensor(model_data);

        create_istft_context(model_data);
        
        // Step 5: Allocate tensors in backend (includes our noise tensor now)
        model_data.buffer = ggml_backend_alloc_ctx_tensors(model_data.ctx, model_data.backend);
        
        // Step 6: Copy tensor data from temp context to backend
        printf("S3GenDataLoader: Copying tensors to backend...\n");
        for (struct ggml_tensor* cur = ggml_get_first_tensor(model_data.ctx); 
             cur != NULL; 
             cur = ggml_get_next_tensor(model_data.ctx, cur)) {
            struct ggml_tensor* src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
            if (src) {  // Only copy if tensor exists in source (noise tensor won't exist there)
                size_t n_size = ggml_nbytes(src);
                ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
            }
        }
        
        model_data.num_tensors = static_cast<uint32_t>(num_tensors);
        model_data.loaded = true;
        
        printf("S3GenDataLoader: Model loaded successfully with backend!\n");
        model_data.print_info();
        
    } catch (const std::exception& e) {
        gguf_free(gguf_ctx);
        ggml_free(tmp_ctx);
        model_data.free();
        throw std::runtime_error("Error loading model with backend: " + std::string(e.what()));
    }
    
    // Cleanup temporary context
    gguf_free(gguf_ctx);
    ggml_free(tmp_ctx);
    
    // Organize F0 predictor tensors
    organize_f0_predictor(model_data);
    
    // Organize source module tensors (noise tensor already created and allocated)
    organize_source_module(model_data);
    
    // Fill noise tensor with Gaussian random values (after backend allocation)
    fill_gaussian_noise_tensor(model_data);

    fill_istft_context(model_data);

    // organize decode module
    organize_decode_module(model_data);

    return model_data;
}

// Load complete S3Gen input data
S3GenInputData S3GenDataLoader::load_s3gen_inputs(const std::string& model_path, const std::string& mels_path, bool full_load) {
    S3GenInputData input_data;
    
    printf("=== Loading S3Gen Input Data ===\n");
    
    // Load mel-spectrogram data
    input_data.mels = load_mels_from_npy(mels_path);
    
    // Load model (metadata only or full with backend)
    if (full_load) {
        printf("Loading full model with backend...\n");
        input_data.model = load_model_with_backend(model_path);
    } else {
        printf("Loading model metadata only...\n");
        input_data.model = load_model_metadata(model_path);
    }
    
    // Print summary
    input_data.print_summary();
    
    return input_data;
}

// Helper functions
void S3GenDataLoader::read_bytes(std::ifstream& file, char* buffer, size_t size) {
    file.read(buffer, size);
    if (!file.good()) {
        throw std::runtime_error("Failed to read bytes from file");
    }
}

uint32_t S3GenDataLoader::read_uint32(std::ifstream& file) {
    uint32_t value;
    read_bytes(file, reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

uint16_t S3GenDataLoader::read_uint16(std::ifstream& file) {
    uint16_t value;
    read_bytes(file, reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

void S3GenDataLoader::validate_npy_header(std::ifstream& file, std::vector<uint32_t>& shape) {
    // Read NPY magic string
    char magic[6];
    read_bytes(file, magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid NPY file: bad magic number");
    }
    
    // Read version
    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);
    
    if (major_version != 1 || minor_version != 0) {
        throw std::runtime_error("Unsupported NPY version");
    }
    
    // Read header length
    uint16_t header_len = read_uint16(file);
    
    // Read and parse header dictionary
    std::string header(header_len, ' ');
    read_bytes(file, &header[0], header_len);
    
    // Simple parsing - look for shape tuple
    size_t shape_start = header.find("'shape': (");
    if (shape_start == std::string::npos) {
        throw std::runtime_error("Could not find shape in NPY header");
    }
    shape_start += 10; // length of "'shape': ("
    
    size_t shape_end = header.find(")", shape_start);
    if (shape_end == std::string::npos) {
        throw std::runtime_error("Malformed shape in NPY header");
    }
    
    // Parse shape dimensions
    std::string shape_str = header.substr(shape_start, shape_end - shape_start);
    
    // Simple parser for comma-separated integers
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.length()) {
        size_t comma_pos = shape_str.find(',', pos);
        if (comma_pos == std::string::npos) {
            comma_pos = shape_str.length();
        }
        
        std::string dim_str = shape_str.substr(pos, comma_pos - pos);
        
        // Remove whitespace
        size_t start = dim_str.find_first_not_of(" \t");
        size_t end = dim_str.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            dim_str = dim_str.substr(start, end - start + 1);
            if (!dim_str.empty() && std::isdigit(dim_str[0])) {
                shape.push_back(std::stoul(dim_str));
            }
        }
        
        pos = comma_pos + 1;
    }
    
    if (shape.empty()) {
        throw std::runtime_error("Could not parse shape from NPY header");
    }
    
    // Verify this is float32 data
    if (header.find("'<f4'") == std::string::npos && header.find("'float32'") == std::string::npos) {
        throw std::runtime_error("NPY file must contain float32 data");
    }
}

// ============================================================================
// HiFT Vocoder (from s3gen_hift.cpp)
// ============================================================================

#include "ggml.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Debug controls - const to satisfy SonarQube; modify locally during debugging
static const bool do_exit = false;
static const int exit_value = 5;
static const int exind = 0;
#define exc(i, t_name) if (do_exit) { return; } else if (i == exit_value) {return t_name;}


// Helper: Generate sine waves from F0 with harmonics (SineGen)  
// f0: [length] - upsampled F0 in Hz
// Returns: [length, harmonic_num+1] - sine waves for each harmonic
struct ggml_tensor* generate_sine_harmonics(
    struct ggml_context* ctx,
    struct ggml_tensor* f0,
    int harmonic_num,
    float sampling_rate,
    float sine_amp,
    float noise_std,
    float voiced_threshold,
    const S3GenModelData& model
) {
    int length = static_cast<int>(f0->ne[0]);  
    
    // ===== STEP 1: Generate normalized harmonic frequencies =====
    // F_mat[i] = (f0 * (i+1) / sampling_rate) % 1.0
    // This gives us normalized frequencies in range [0, 1)
    
    std::vector<struct ggml_tensor*> harmonics;
    
    for (int i = 0; i < harmonic_num + 1; i++) {
        float harmonic_scale = (i + 1) / sampling_rate;
        
        // Scale f0 by (i+1)/sampling_rate to get normalized frequency
        struct ggml_tensor* harmonic = ggml_scale(ctx, f0, harmonic_scale);
        
        // Apply modulo 1.0 using: x % 1.0 = x - floor(x)
        // This wraps frequencies to [0, 1) range, preventing overflow
        struct ggml_tensor* harmonic_floor = ggml_floor(ctx, harmonic);
        harmonic = ggml_sub(ctx, harmonic, harmonic_floor);
        
        // Reshape to [length, 1] for concatenation
        harmonic = ggml_reshape_2d(ctx, harmonic, length, 1);
        
        harmonics.push_back(harmonic);
    }
    
    // Concatenate all harmonics: [length, harmonic_num+1]
    struct ggml_tensor* F_mat = harmonics[0];
    for (int i = 1; i < harmonic_num + 1; i++) {
        F_mat = ggml_concat(ctx, F_mat, harmonics[i], 1);
    }
    
    // ===== STEP 3: Compute cumulative phase =====
    // phase = cumsum(F_mat, dim=0) % 1.0
    struct ggml_tensor* phase = ggml_cumsum(ctx, F_mat);
    // Apply modulo 1.0 after cumsum to prevent phase wraparound
    // phase = phase - floor(phase)
    struct ggml_tensor* phase_floor = ggml_floor(ctx, phase);
    phase = ggml_sub(ctx, phase, phase_floor);
    // ===== STEP 4: Convert to radians and generate sine =====
    // theta = 2*pi * phase
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    struct ggml_tensor* theta = ggml_scale(ctx, phase, two_pi);
    
    // ===== STEP 5: Create random phase vector =====
    // Generate random phase offsets for each harmonic
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-M_PI, M_PI);
    
    // First harmonic (fundamental): phase = 0
    struct ggml_tensor *fundamental = ggml_arange(ctx, 0.0f, 1.0f, 1.0f);  // Creates [0]
    
    // Other harmonics: random phases [-π, π]
    for(int i = 1; i < harmonic_num + 1; i++) {
        float random = static_cast<float>(dis(gen));
        struct ggml_tensor *phase_scalar = ggml_arange(ctx, random, random+1, 10.0f);  // Creates [random]
        fundamental = ggml_concat(ctx, fundamental, phase_scalar, 0);
    }
    
    // Step 3: ggml_reshape -> (1, 9, 1) 
    struct ggml_tensor *phase_vec_3d = ggml_reshape_3d(ctx, fundamental, 1, harmonic_num + 1, 1);
 
    // Step 4: ggml_repeat -> (length, 9, 1) to match theta dimensions
    struct ggml_tensor *phase_vec_broadcast = ggml_repeat(ctx, phase_vec_3d, theta);

    // Step 5: ggml_add - Add random phases to theta
    struct ggml_tensor *theta_with_phases = ggml_add(ctx, theta, phase_vec_broadcast);
    // ===== STEP 6: Generate sine waves with random phases =====
    // sine_waves = sin(theta_with_phases)
    struct ggml_tensor *sine_waves = ggml_sin(ctx, theta);
    // // Scale by amplitude
    sine_waves = ggml_scale(ctx, sine_waves, sine_amp);

    // ===== STEP 7: Generate UV (voiced/unvoiced) mask =====
    // PyTorch: uv = (f0 > self.voiced_threshold).type(torch.float32)
    // Create threshold tensor and broadcast to f0 shape
    struct ggml_tensor* threshold_tensor = ggml_arange(ctx, voiced_threshold, voiced_threshold+1, 10.0f);
    struct ggml_tensor* threshold_broadcast = ggml_repeat(ctx, threshold_tensor, f0);
     
    // Compute f0 - threshold, then apply step function
    // Since step(x) = 1 if x >= 0, we use (f0 - threshold + epsilon) to get f0 > threshold
    struct ggml_tensor* f0_diff = ggml_sub(ctx, f0, threshold_broadcast);
    struct ggml_tensor* epsilon_tensor = ggml_arange(ctx, 1e-6f, 1e-6f+1, 10.0f);  // Small positive value
    struct ggml_tensor* epsilon_broadcast = ggml_repeat(ctx, epsilon_tensor, f0);
    f0_diff = ggml_sub(ctx, f0_diff, epsilon_broadcast);  // f0 - threshold - epsilon
    struct ggml_tensor* uv = ggml_step(ctx, f0_diff);  // 1 if f0 > threshold, 0 otherwise
     
    //  ===== STEP 8: Generate noise amplitude =====
    //  PyTorch: noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
     
    //  Broadcast UV to match sine_waves shape [length, harmonic_num+1]
    struct ggml_tensor* uv_broadcast = ggml_repeat(ctx, 
        ggml_reshape_2d(ctx, uv, length, 1), f0);
     
    // Create noise_std and sine_amp/3 tensors
    struct ggml_tensor* noise_std_tensor = ggml_arange(ctx, noise_std, noise_std+1, 10.0f);
    struct ggml_tensor* sine_amp_div3_tensor = ggml_arange(ctx, sine_amp / 3.0f, sine_amp / 3.0f+1, 10.0f);
     
    // Broadcast to match sine_waves shape
    struct ggml_tensor* noise_std_broadcast = ggml_repeat(ctx, noise_std_tensor, f0);
    struct ggml_tensor* sine_amp_div3_broadcast = ggml_repeat(ctx, sine_amp_div3_tensor, f0);
     
    // Calculate (1 - uv)
    struct ggml_tensor* one_tensor = ggml_arange(ctx, 1.0f, 1.0f+1, 10.0f);
    struct ggml_tensor* one_broadcast = ggml_repeat(ctx, one_tensor, f0);
    struct ggml_tensor* one_minus_uv = ggml_sub(ctx, one_broadcast, uv_broadcast);
     
    // Calculate noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    struct ggml_tensor* uv_noise_part = ggml_mul(ctx, uv_broadcast, noise_std_broadcast);
    struct ggml_tensor* unvoiced_noise_part = ggml_mul(ctx, one_minus_uv, sine_amp_div3_broadcast);
    struct ggml_tensor* noise_amp = ggml_add(ctx, uv_noise_part, unvoiced_noise_part);
    // ===== STEP 9: Use pre-allocated Gaussian noise tensor =====
    // PyTorch: noise = noise_amp * torch.randn_like(sine_waves)
    // Use the pre-allocated Gaussian noise tensor from the model (already filled during model loading)
     
    // Apply noise amplitude: noise = noise_amp * randn_tensor
    // Need to broadcast noise_amp from [length] to [length, harmonic_num+1] to match noise tensor
    struct ggml_tensor* noise_amp_broadcast = ggml_repeat(ctx, 
         ggml_reshape_2d(ctx, noise_amp, noise_amp->ne[0], 1), sine_waves);
     
    // Use the pre-allocated Gaussian noise tensor from the model
    struct ggml_tensor* noise = ggml_mul(ctx, noise_amp_broadcast, model.source.noise_tensor);
    // ===== STEP 10: Apply UV masking =====
    // PyTorch: sine_waves = sine_waves * uv + noise
    // Need to broadcast uv_broadcast from [length] to [length, harmonic_num+1] to match sine_waves
     
    struct ggml_tensor* uv_sine_broadcast = ggml_repeat(ctx, 
        ggml_reshape_2d(ctx, uv_broadcast, uv_broadcast->ne[0], 1), sine_waves);
     
    struct ggml_tensor* voiced_part = ggml_mul(ctx, sine_waves, uv_sine_broadcast);
    struct ggml_tensor* final_waves = ggml_add(ctx, voiced_part, noise);
     
    return final_waves;
}

// Create STFT placeholder - zero tensor with STFT output shape
// Input: s [time_wav, 1, batch] 
// Output: s_stft [time_frames, n_fft+2, batch]
// Where time_frames = 1 + time_wav // hop_len
struct ggml_tensor* create_stft_placeholder(
    struct ggml_context* ctx,
    struct ggml_tensor* s,
    int n_fft,
    int hop_len
) {
    // Calculate STFT dimensions
    // s shape: [time_wav, 1, batch] in GGML format
    int64_t time_wav = s->ne[0];
    int64_t batch = s->ne[2];
    
    // Calculate number of STFT frames (with center=True by default)
    int64_t time_frames = 1 + time_wav / hop_len;
    
    // STFT produces n_fft/2 + 1 frequency bins = 9 for n_fft=16
    // We concatenate real and imaginary parts: 9 + 9 = 18 channels
    int64_t n_channels = n_fft + 2;  // 18 for n_fft=16
    
    // Create zero tensor [time_frames, n_channels, batch] using arange + repeat
    // Step 1: Create a scalar zero tensor using ggml_arange
    struct ggml_tensor* zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 10.0f);  // Creates [0.0]
    
    // Step 2: Reshape to [1, 1, 1]
    struct ggml_tensor* zero_3d = ggml_reshape_3d(ctx, zero_scalar, 1, 1, 1);
    
    // Step 3: Create target shape tensor [time_frames, n_channels, batch]
    // We need to repeat the zero across all dimensions
    // ggml_repeat broadcasts the input to match the target shape
    // First create a dummy tensor with the target shape to use as reference
    struct ggml_tensor* target_shape = ggml_arange(ctx, 0.0f, (float)(time_frames * n_channels * batch), 1.0f);
    target_shape = ggml_reshape_3d(ctx, target_shape, time_frames, n_channels, batch);
    
    // Step 4: Repeat zero_3d to match target shape
    return ggml_repeat(ctx, zero_3d, target_shape);
}

// Apply Snake activation: x + (1/alpha) * sin^2(alpha * x)


// Apply a single ResBlock (residual block with dilated convolutions)
struct ggml_tensor* apply_resblock(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    const resblock& block,
    int kernel_size
) {
    struct ggml_tensor* residual = x;
    
    // Apply 3 dilation groups
    const int dilations[3] = {1, 3, 5};
    
    for (int i = 0; i < 3; i++) {
        // Calculate padding: padding = dilation * (kernel_size - 1) / 2
        int padding1 = dilations[i] * (kernel_size - 1) / 2;
        
        // First convolution path (convs1)
        struct ggml_tensor* xt = apply_snake_activation(ctx, residual, block.activations_alpha_1[i]);
        
        struct ggml_tensor* weight1 = apply_weight_norm(ctx, block.layer_group_1[i].weight_v, 
                                                        block.layer_group_1[i].weight_g);
        xt = apply_conv1d_nobatch(ctx, xt, weight1, block.layer_group_1[i].bias, 1, padding1, dilations[i]);
        
        // Second convolution path (convs2)
        xt = apply_snake_activation(ctx, xt, block.activation_alpha_2[i]);
        
        struct ggml_tensor* weight2 = apply_weight_norm(ctx, block.layer_group_2[i].weight_v,
                                                        block.layer_group_2[i].weight_g);
        int padding2 = (kernel_size - 1) / 2;  // dilation=1 for convs2
        xt = apply_conv1d_nobatch(ctx, xt, weight2, block.layer_group_2[i].bias, 1, padding2, 1);
        
        // Residual connection: x = xt + x
        residual = ggml_add(ctx, xt, residual);
    }
    
    return residual;
}

// Apply a source resblock (similar structure to regular resblock)
struct ggml_tensor* apply_source_resblock(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    const source_resblock& block,
    int kernel_size
) {
    struct ggml_tensor* residual = x;
    
    const int dilations[3] = {1, 3, 5};
    
    for (int i = 0; i < 3; i++) {
        // Calculate padding: padding = dilation * (kernel_size - 1) / 2
        int padding1 = dilations[i] * (kernel_size - 1) / 2;
        
        struct ggml_tensor* xt = apply_snake_activation(ctx, residual, block.activations_alpha_1[i]);
        
        struct ggml_tensor* weight1 = apply_weight_norm(ctx, block.layer_group_1[i].weight_v,
                                                        block.layer_group_1[i].weight_g);
        xt = apply_conv1d_nobatch(ctx, xt, weight1, block.layer_group_1[i].bias, 1, padding1, dilations[i]);
        
        xt = apply_snake_activation(ctx, xt, block.activation_alpha_2[i]);
        
        struct ggml_tensor* weight2 = apply_weight_norm(ctx, block.layer_group_2[i].weight_v,
                                                        block.layer_group_2[i].weight_g);
        int padding2 = (kernel_size - 1) / 2;  // dilation=1 for convs2
        xt = apply_conv1d_nobatch(ctx, xt, weight2, block.layer_group_2[i].bias, 1, padding2, 1);
        
        residual = ggml_add(ctx, xt, residual);
    }
    
    return residual;
}

// Apply conv_pre layer
// Input: x [time_mel, in_channels, batch] = [T_mel, 80, 1]
// Output: [time_mel, base_channels, batch] = [T_mel, 512, 1]
struct ggml_tensor* apply_conv_pre(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    const layer& conv_pre_layer
) {
    // Apply weight normalization: weight = weight_g * (weight_v / ||weight_v||)
    struct ggml_tensor* weight = apply_weight_norm(ctx, conv_pre_layer.weight_v, conv_pre_layer.weight_g);
    
    // Conv1d parameters for conv_pre: kernel_size=7, stride=1, padding=3
    int stride = 1;
    int padding = 3;
    int dilation = 1;
    
    // Apply convolution with bias
    return apply_conv1d_nobatch(ctx, x, weight, conv_pre_layer.bias, stride, padding, dilation);
}

struct ggml_tensor* apply_conv_post(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    const layer& conv_post_layer
) {
    struct ggml_tensor* weight = apply_weight_norm(ctx, conv_post_layer.weight_v, conv_post_layer.weight_g);

    return apply_conv1d_nobatch(ctx, x, weight, conv_post_layer.bias, 1, 3, 1);
}

// Manual ISTFT implementation matching torch.istft with center=True
// magnitude: [time_frames, n_fft/2+1, batch] = [10320, 9, 1]
// phase: [time_frames, n_fft/2+1, batch] = [10320, 9, 1]
// Returns: [time_samples, 1, batch] audio waveform
struct ggml_tensor* istft(
    struct ggml_context* ctx,
    struct ggml_tensor* magnitude,
    struct ggml_tensor* phase,
    const S3GenModelData& model
) {
    // ISTFT parameters (matching Python implementation)
    const int n_fft = 16;
    const int hop_len = 4;
    const int win_length = n_fft;
    const float max_clip = 1e2f;
    
    // Input shape: [time_frames, num_bins, batch]
    int64_t batch_size = magnitude->ne[2];
    int64_t num_bins = magnitude->ne[1];  // Should be 9 (n_fft/2 + 1)
    int64_t num_frames = magnitude->ne[0];
    
    printf("ISTFT: num_frames=%lld, num_bins=%lld, batch=%lld\n", num_frames, num_bins, batch_size);
    
    // ===== STEP 1: Clip magnitude =====
    // magnitude = torch.clip(magnitude, max=1e2)
    struct ggml_tensor* mag_clipped = ggml_clamp(ctx, magnitude, -INFINITY, max_clip);
    
    // ===== STEP 2: Create complex spectrogram =====
    // real = magnitude * cos(phase)
    // imag = magnitude * sin(phase)
    struct ggml_tensor* cos_phase = ggml_cos(ctx, phase);
    struct ggml_tensor* sin_phase = ggml_sin(ctx, phase);
    
    struct ggml_tensor* real = ggml_mul(ctx, mag_clipped, cos_phase);
    struct ggml_tensor* imag = ggml_mul(ctx, mag_clipped, sin_phase);
    
    // ===== STEP 3: Set DC and Nyquist imaginary components to 0 =====
    // imag[:, 0, :] = 0.0  (DC component)
    // imag[:, -1, :] = 0.0 (Nyquist component if present)
    
    // Create zero tensor for DC component
    struct ggml_tensor* zero_scalar = ggml_arange(ctx, 0.0f, 1.0f, 10.0f);
    struct ggml_tensor* zero_dc = ggml_repeat(ctx, 
        ggml_reshape_3d(ctx, zero_scalar, 1, 1, 1),
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, num_frames, 1, batch_size)
    );
    
    // Use ggml_set_1d to set DC bin to zero
    // Extract view of DC bin (channel 0)
    struct ggml_tensor* imag_dc = ggml_view_3d(ctx, imag,
        num_frames, 1, batch_size,
        imag->nb[1], imag->nb[2], 0
    );
    
    // Set DC to zero by creating new imag with DC replaced
    // This is tricky - we need to create views and concatenate
    // For now, let's assume DC is already small and skip explicit zeroing
    // (The model should naturally produce small values there)
    
    // ===== STEP 4: Reconstruct full spectrum with conjugate symmetry =====
    // For n_fft=16 (even), num_bins=9: we have [DC, f1, ..., f7, Nyquist]
    // Need to mirror [f7, f6, ..., f1] to get full 16-point spectrum
    // Exclude DC (index 0) and Nyquist (index 8): mirror indices [7,6,5,4,3,2,1]
    
    // Extract middle frequencies (excluding DC and Nyquist): indices 1 to 7
    // real_middle = real[:, 1:-1, :]
    struct ggml_tensor* real_middle = ggml_view_3d(ctx, real,
        num_frames, num_bins - 2, batch_size,  // 7 channels
        real->nb[1], real->nb[2], 
        real->nb[1]  // Offset by 1 channel
    );
    
    struct ggml_tensor* imag_middle = ggml_view_3d(ctx, imag,
        num_frames, num_bins - 2, batch_size,
        imag->nb[1], imag->nb[2],
        imag->nb[1]
    );
    
    // Flip along channel dimension (dim 1)
    // GGML doesn't have a flip operation, so we'll use views to extract in reverse order
    std::vector<struct ggml_tensor*> real_flipped_channels;
    std::vector<struct ggml_tensor*> imag_flipped_channels;
    
    for (int64_t i = num_bins - 3; i >= 0; i--) {  // Indices 7,6,5,4,3,2,1 -> 6,5,4,3,2,1,0 in real_middle
        struct ggml_tensor* real_ch = ggml_view_3d(ctx, real_middle,
            num_frames, 1, batch_size,
            real_middle->nb[1], real_middle->nb[2],
            i * real_middle->nb[1]
        );
        real_flipped_channels.push_back(real_ch);
        
        // For imaginary, we need to negate (conjugate)
        struct ggml_tensor* imag_ch = ggml_view_3d(ctx, imag_middle,
            num_frames, 1, batch_size,
            imag_middle->nb[1], imag_middle->nb[2],
            i * imag_middle->nb[1]
        );
        // Negate: -imag (use dup+scale since imag_ch is a view)
        imag_ch = ggml_dup(ctx, imag_ch);
        imag_ch = ggml_scale(ctx, imag_ch, -1.0f);
        imag_flipped_channels.push_back(imag_ch);
    }
    
    // Concatenate flipped channels
    struct ggml_tensor* real_mirror = real_flipped_channels[0];
    struct ggml_tensor* imag_mirror = imag_flipped_channels[0];
    
    for (size_t i = 1; i < real_flipped_channels.size(); i++) {
        real_mirror = ggml_concat(ctx, real_mirror, real_flipped_channels[i], 1);
        imag_mirror = ggml_concat(ctx, imag_mirror, imag_flipped_channels[i], 1);
    }
    
    // Concatenate: [DC, f1..f7, Nyquist, f7..f1_mirror]
    // real_full = torch.cat([real, real_mirror], dim=1)
    struct ggml_tensor* real_full = ggml_concat(ctx, real, real_mirror, 1);
    struct ggml_tensor* imag_full = ggml_concat(ctx, imag, imag_mirror, 1);
    
    printf("ISTFT: real_full shape = [%d, %d, %d]\n", 
           (int)real_full->ne[0], (int)real_full->ne[1], (int)real_full->ne[2]);
    
    // ===== STEP 5: Manual IFFT using DFT matrix =====
    // Create twiddle factors for IFFT
    // angles = 2π * k * n / N for k,n in [0, n_fft)
    // DFT matrices are pre-loaded from model weights
    // (cos_matrix and sin_matrix tensors loaded during model initialization)
    
    // Get GGML tensors for DFT matrices [n_fft, n_fft]
    struct ggml_tensor* cos_matrix = model.istft.cos_matrix;
    struct ggml_tensor* sin_matrix = model.istft.sin_matrix;
    
    // ===== STEP 6: Apply IFFT to each frame =====
    // We need to loop over frames and apply matrix multiplication
    // This is challenging in GGML's graph model - we'll create a batched operation
    
    // Reshape spectrograms for batched matmul
    // real_full: [num_frames, n_fft, batch] -> [n_fft, num_frames * batch]
    // We want to do: time_signal[n, frame] = sum_k(spec[k, frame] * W[k, n])
    
    // Transpose real_full and imag_full: [num_frames, n_fft, batch] -> [n_fft, num_frames, batch]
    struct ggml_tensor* real_transposed = ggml_cont(ctx, ggml_permute(ctx, real_full, 1, 0, 2, 3));
    struct ggml_tensor* imag_transposed = ggml_cont(ctx, ggml_permute(ctx, imag_full, 1, 0, 2, 3));
    
    // Reshape to [n_fft, num_frames * batch]
    struct ggml_tensor* real_2d = ggml_reshape_2d(ctx, real_transposed, 
        n_fft, num_frames * batch_size);
    struct ggml_tensor* imag_2d = ggml_reshape_2d(ctx, imag_transposed,
        n_fft, num_frames * batch_size);
    
    // Matrix multiply: cos_matrix @ real_2d - sin_matrix @ imag_2d
    // cos_matrix: [n_fft, n_fft], real_2d: [n_fft, num_frames * batch]
    // Result: [n_fft, num_frames * batch]
    struct ggml_tensor* real_part = ggml_mul_mat(ctx, cos_matrix, real_2d);
    struct ggml_tensor* imag_part = ggml_mul_mat(ctx, sin_matrix, imag_2d);
    struct ggml_tensor* time_frames = ggml_sub(ctx, real_part, imag_part);
    
    // Normalize by N (IFFT scaling)
    time_frames = ggml_scale(ctx, time_frames, 1.0f / n_fft);
    
    // Reshape back: [n_fft, num_frames * batch] -> [n_fft, num_frames, batch]
    time_frames = ggml_reshape_3d(ctx, time_frames, n_fft, num_frames, batch_size);
    
    // Transpose: [n_fft, num_frames, batch] -> [num_frames, n_fft, batch]
    time_frames = ggml_cont(ctx, ggml_permute(ctx, time_frames, 1, 0, 2, 3));
    
    printf("ISTFT: time_frames shape = [%d, %d, %d]\n",
           (int)time_frames->ne[0], (int)time_frames->ne[1], (int)time_frames->ne[2]);
    
    // ===== STEP 7: Create window function =====
    // Hann window: w[n] = 0.5 * (1 - cos(2π * n / (N-1)))
    // std::vector<float> window_data(win_length);
    // for (int i = 0; i < win_length; i++) {
    //     window_data[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (win_length - 1)));
    // }
    
    struct ggml_tensor* window = model.istft.hann_window;
    
    // // Fill window with data if context has allocated memory
    // if (window->data != nullptr) {
    //     float* window_ptr = (float*)window->data;
    //     for (size_t i = 0; i < window_data.size(); i++) {
    //         window_ptr[i] = window_data[i];
    //     }
    // }
    
    // ===== STEP 8: Overlap-add reconstruction =====
    // Calculate output length
    int64_t expected_length = hop_len * (num_frames - 1);
    int64_t padded_length = expected_length + n_fft;
    
    printf("ISTFT: expected_length=%d, padded_length=%d\n", (int)expected_length, (int)padded_length);
    
    // ===== STEP 9: Overlap-add reconstruction =====
    // Use custom GGML operation for efficient overlap-add with proper window normalization
    // This operation handles windowing, overlap-add, and window sum normalization
    
    printf("ISTFT: Performing overlap-add with custom GGML operation\n");
    
    // Apply overlap-add: time_frames [num_frames, n_fft, batch] + window [n_fft] 
    //                  -> output [expected_length, batch]
    // The operation will:
    // 1. Apply window to each frame
    // 2. Overlap-add the windowed frames
    // 3. Accumulate window overlap sum
    // 4. Normalize by dividing signal by window_sum
    struct ggml_tensor* overlap_added = ggml_overlap_add(ctx, time_frames, window, hop_len, n_fft / 2);
    
    printf("ISTFT: Overlap-add output shape = [%d, %d]\n",
           (int)overlap_added->ne[0], (int)overlap_added->ne[1]);
    
    // Reshape to match expected output format [time_samples, 1, batch]
    struct ggml_tensor* output = ggml_reshape_3d(ctx, overlap_added, 
        overlap_added->ne[0], 1, overlap_added->ne[1]);
    
    printf("ISTFT: Complete! Output shape = [%d, %d, %d]\n",
           (int)output->ne[0], (int)output->ne[1], (int)output->ne[2]);
    
    return output;
}
// Decode function: Convert mel-spectrogram and source signal to audio waveform
// mels: [time_mel, 80, batch] - mel-spectrogram features
// source: [time_wav, 1, batch] - harmonic source signal from apply_source_module
// Returns: audio waveform [time_wav] 
struct ggml_tensor* decode(
    struct ggml_context* ctx,
    struct ggml_tensor* mels,
    // struct ggml_tensor* source,
    const S3GenModelData& model
) {
    // STFT parameters (from hifigan.py line 301)
    const int n_fft = 16;
    const int hop_len = 4;
    
    // Step 1: Create STFT placeholder (zero tensor)
    // In PyTorch: s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
    //             s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
    // For now, we create a zero tensor with the expected shape
    // struct ggml_tensor* s_stft = create_stft_placeholder(ctx, source, n_fft, hop_len);
    struct ggml_tensor* target_shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mels->ne[0]*120+1, 18, 1, 1);
    struct ggml_tensor* s_stft = ggml_arange(ctx, 0.0f, 1.0f, 10.0f);
    s_stft = ggml_repeat(ctx, s_stft, target_shape);
    
    // Step 2: Apply conv_pre layer
    struct ggml_tensor* x = apply_conv_pre(ctx, mels, model.decode.pre_conv_layer);
    
    // Step 3: Upsampling loop (3 stages with 256× total upsampling)
    const int num_upsamples = 3;
    const int num_kernels = 3;  // resblock_kernel_sizes=[3, 7, 11]
    const float lrelu_slope = 0.1f;
    
    for (int i = 0; i < num_upsamples; i++) {
        // 3.1: LeakyReLU activation
        x = ggml_leaky_relu(ctx, x, lrelu_slope, true);
        
        // 3.2: Transposed convolution (upsampling)
        // Apply weight normalization
        struct ggml_tensor* up_weight = apply_weight_norm(
            ctx,
            model.decode.up_layers[i].weight_v,
            model.decode.up_layers[i].weight_g
        );
        
        // Get upsample parameters
        // Based on loaded model: 3 up layers with different parameters
        // m2wsups_0: stride=8, kernel=16, padding=4
        // m2wsups_1: stride=5, kernel=11, padding=3
        // m2wsups_2: stride=3, kernel=7, padding=2
        // Total upsampling: 8 × 5 × 3 = 120× (with ISTFT hop_len=4: 480× total)
        const int upsample_rates[3] = {8, 5, 3};
        const int upsample_kernel_sizes[3] = {16, 11, 7};
        int stride = upsample_rates[i];
        int kernel_size = upsample_kernel_sizes[i];
        int padding = (kernel_size - stride) / 2;
        
        // Apply transposed convolution
        ggml_tensor* weight_f32 = ggml_cast(ctx, up_weight, GGML_TYPE_F32);
        x = ggml_conv_transpose_1d(ctx, weight_f32, x, stride, padding, 1);
        
        // Add bias
        struct ggml_tensor* bias_reshaped = ggml_reshape_4d(ctx, model.decode.up_layers[i].bias, 
                                                            1, model.decode.up_layers[i].bias->ne[0], 1, 1);
        x = ggml_add(ctx, x, bias_reshaped);
        // 3.3: Reflection padding (only on last iteration)
        if (i == num_upsamples - 1) {
            // Implement reflection padding left=1 using view and concat
            // Original: [a, b, c, d, ...] → Reflected: [b, a, b, c, d, ...]
            // Extract element at index 1 (second element)
            struct ggml_tensor* reflected_elem = ggml_view_3d(
                ctx,
                x,
                1,              // Extract 1 time step
                x->ne[1],       // All channels
                x->ne[2],       // All batch
                x->nb[1],       // Stride for dimension 0
                x->nb[2],       // Stride for dimension 1
                x->nb[1]        // Offset by 1 element (to get index 1)
            );
            
            // Concatenate: [reflected_elem, original_x] along dimension 0 (time)
            x = ggml_concat(ctx, reflected_elem, x, 0);
        }
        // 3.4: Source fusion
        
        int down_strides[3] = {15, 3, 1};
        int down_paddings[3] = {7, 1, 0};
        
        // The third down layer weight needs permutation from [18,64,1,1] to [1,18,64,1]
        struct ggml_tensor* down_weight = model.decode.down_layers[i].weight_v;
        if (i == 2) {
            // Permute: [18, 64, 1, 1] -> [1, 18, 64, 1]
            down_weight = ggml_cont(ctx, ggml_permute(ctx, down_weight, 1, 2, 3, 0));

        }
        if (down_weight->type != GGML_TYPE_F32) {
            down_weight = ggml_cast(ctx, down_weight, GGML_TYPE_F32);
        }
        struct ggml_tensor* si = ggml_conv_1d(
            ctx,
            down_weight,
            s_stft,
            down_strides[i],
            down_paddings[i],
            1  // dilation
        );
        // Add bias
        struct ggml_tensor* si_bias_reshaped = ggml_reshape_4d(ctx, model.decode.down_layers[i].bias,
            1, model.decode.down_layers[i].bias->ne[0], 1, 1);
        si = ggml_add(ctx, si, si_bias_reshaped);
        
        // Apply source resblock with appropriate kernel size
        // source_resblock kernel sizes: [7, 7, 11]
        const int source_kernel_sizes[3] = {7, 7, 11};
        si = apply_source_resblock(ctx, si, model.decode.source_resblocks[i], source_kernel_sizes[i]);
        // Fusion: add source features to main path
        x = ggml_add(ctx, x, si);
        // 3.5: Multi-kernel residual blocks (3 parallel ResBlocks with different kernels)
        // resblock_kernel_sizes = [3, 7, 11]
        const int resblock_kernel_sizes[3] = {3, 7, 11};
        struct ggml_tensor* xs = nullptr;
        for (int j = 0; j < num_kernels; j++) {
            int resblock_idx = i * num_kernels + j;
            struct ggml_tensor* xj = apply_resblock(ctx, x, model.decode.resblocks[resblock_idx], 
                                                    resblock_kernel_sizes[j]);
            
            if (xs == nullptr) {
                xs = xj;
            } else {
                xs = ggml_add(ctx, xs, xj);
            }
        }
        
        // Average the outputs from 3 ResBlocks
        x = ggml_scale(ctx, xs, 1.0f / num_kernels);
    }
    
    x = ggml_leaky_relu(ctx, x, 0.01f, true);
    
    x = apply_conv_post(ctx, x, model.decode.post_conv_layer);
    
    // Magnitude/phase extraction
    // x shape: [batch, 18, time] where 18 = n_fft + 2
    // Split into magnitude (first 9 channels) and phase (last 9 channels)
    // n_fft = 16, so n_fft // 2 + 1 = 9
    const int mag_channels = n_fft / 2 + 1;  // 9 channels
    
    // Extract magnitude: first 9 channels
    // ggml_view_3d to extract channels 0-8
    struct ggml_tensor* magnitude = ggml_view_3d(
        ctx,
        x,
        x->ne[0],           // time dimension
        mag_channels,       // first 9 channels
        x->ne[2],           // batch dimension
        x->nb[1],           // stride in bytes for ne[0]
        x->nb[2],           // stride in bytes for ne[1]
        0                   // offset = 0 (start from first channel)
    );
    magnitude = ggml_exp(ctx, magnitude);  // Apply exp
    
    // // Extract phase: last 9 channels  
    struct ggml_tensor* phase = ggml_view_3d(
        ctx,
        x,
        x->ne[0],                    // time dimension
        mag_channels,                // last 9 channels
        x->ne[2],                    // batch dimension
        x->nb[1],                    // stride in bytes for ne[0]
        x->nb[2],                    // stride in bytes for ne[1]
        mag_channels * x->nb[1]      // offset = 9 channels
    );
    phase = ggml_sin(ctx, phase);  // Apply sin
    
    // TODO: Implement ISTFT to convert magnitude/phase to audio
    x = istft(ctx, magnitude, phase, model);
    return x;  // Temporary return - will be audio waveform when ISTFT is complete
}

// Apply Neural Source Filter (SourceModuleHnNSF)
// Converts F0 to harmonic excitation signal
struct ggml_tensor* apply_source_module(
    struct ggml_context* ctx,
    struct ggml_tensor* f0_upsampled,
    const S3GenModelData& model
) {
    // Parameters from HiFiGAN configuration
    const int harmonic_num = 8;           // nb_harmonics
    const float sampling_rate = 24000.0f; // 24kHz
    const float sine_amp = 0.1f;          // nsf_alpha
    const float noise_std = 0.003f;       // nsf_sigma  
    const float voiced_threshold = 10.0f; // nsf_voiced_threshold
    
    // Step 1: Generate sine harmonics using SineGen
    // Input: [41280] F0 values
    // Output: [41280, 9] sine waves (8 harmonics + fundamental)
    struct ggml_tensor* sine_waves = generate_sine_harmonics(
        ctx, f0_upsampled, harmonic_num, sampling_rate,
        sine_amp, noise_std, voiced_threshold, model
    );
    
    // Step 2: Apply linear layer to merge harmonics
    // weight: [9, 1], bias: [1]
    // Need to transpose sine_waves from [length, harmonics] to [harmonics, length] for matrix multiplication
    // sine_waves: [44160, 9] -> sine_waves_transposed: [9, 44160]
    struct ggml_tensor* sine_waves_transposed = ggml_cont(ctx, ggml_permute(ctx, sine_waves, 1, 0, 2, 3));
    
    struct ggml_tensor* sine_merge = apply_linear(
        ctx,
        sine_waves_transposed,       // [9, 44160]
        model.source.linear_weight,  // [9, 1]
        model.source.linear_bias     // [1]
    );
    // Step 3: Apply tanh activation
    sine_merge = ggml_tanh(ctx, sine_merge);
    
    // Reshape to [41280] for output
    sine_merge = ggml_reshape_1d(ctx, sine_merge, sine_merge->ne[1]);
    
    return sine_merge;
}

ggml_tensor* define_graph(ggml_context* ctx, const S3GenModelData& model, struct ggml_tensor* mel_input) {
    struct ggml_tensor* x = decode(ctx, mel_input, model);
    return x;
}

// Build computation graph for HiFi-GAN inference
HiFTInferenceGraphResult build_hift_inference_graph(
    struct ggml_context* ctx,
    const S3GenModelData& model,
    struct ggml_tensor* mel_input
) {
    // mel_input shape should be: [time_steps, in_channels=80, batch=1]
    // struct ggml_tensor* x = mel_input;
    // struct ggml_tensor* mels = ggml_dup(ctx, mel_input);
    // struct ggml_tensor* test = nullptr;

    // // Apply 5 convolutional layers with ELU activations
    // for (int i = 0; i < 5; i++) {
    //     // Apply weight normalization
    //     struct ggml_tensor* weight = apply_weight_norm(
    //         ctx,
    //         model.f0pred.conv_layers[i].weight_v,
    //         model.f0pred.conv_layers[i].weight_g
    //     );
        
    //     // apply_weight_norm(ctx, model.f0pred.conv_layers[i].weight_v, model.f0pred.conv_layers[i].weight_g);
    //     // Apply Conv1d
    //     x = apply_conv1d_nobatch(
    //         ctx,
    //         x,
    //         weight,
    //         model.f0pred.conv_layers[i].bias,
    //         1,  // stride
    //         1,  // padding
    //         1   // dilation
    //     );
    //     // Apply ELU activation
    //     x = apply_elu(ctx, x);
    // }
    // // Reshape and transpose from [time_steps, out_channels, batch, 1] to [out_channels, time_steps]
    // // Current: [86, 512, 1, 1] -> Need: [512, 86] for linear layer
    // // For linear layer with weight [1, 512], input needs to be [batch*time_steps, in_features] = [86, 512]
    // // Or we can view it as [batch, time_steps, in_features] = [1, 86, 512]
    
    // // First permute to get [out_channels, time_steps, batch, 1]
    // x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    
    // // Reshape to [out_channels, time_steps*batch] = [512, 86]
    // x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    
    // // Apply classifier (Linear layer)
    // x = apply_linear(
    //     ctx,
    //     x,
    //     model.f0pred.classifier_weight,
    //     model.f0pred.classifier_bias
    // );
    
    // // Result from linear is [out_features, seq_len] = [1, 86]
    // // We want [seq_len] = [86] for the final F0 prediction
    // // Reshape to 1D
    // x = ggml_reshape_1d(ctx, x, x->ne[1]);
    
    // // Apply absolute value
    // // Now supported in CUDA backend with our custom kernel
    // x = ggml_abs(ctx, x);
    
    // // Upsample F0 from mel rate to audio sample rate
    // // Scale factor = hop_length = 480
    // int input_len = x->ne[0];
    // int scale_factor = 480;
    // int output_len = input_len * scale_factor;
    // x = ggml_upscale_ext(ctx, x, output_len, 1, 1, 1, GGML_SCALE_MODE_NEAREST);
    
    // // Apply Neural Source Filter (SourceModuleHnNSF)
    // // Generates harmonic excitation signal from F0
    // x = apply_source_module(ctx, x, model);
    
    // // Reshape source from [44160] to [44160, 1, 1] for decode function
    // struct ggml_tensor* source_reshaped = ggml_reshape_3d(ctx, x, x->ne[0], 1, 1);

    // Apply decode (vocoder) - convert mel + source to audio waveform
    // ggml_tensor* x = define_graph(ctx, model, mel_input);
    struct ggml_tensor* x = decode(ctx, mel_input, model);
    
    struct ggml_tensor* test = ggml_dup(ctx, x);
    // Build the computation graph
    // Use larger graph size (8192) to accommodate ISTFT's complex operations
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, test);
    
    HiFTInferenceGraphResult result;
    result.graph = gf;
    result.output = x;
    result.test = test;
    return result;
}
