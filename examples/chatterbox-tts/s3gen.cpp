#include <chrono>
#include <random>
#include "s3gen.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "ggml-cpu.h"


#include "s3Mel2Wav.h"


template <typename T>
std::vector<T> randn_like(size_t size)
{
    std::vector<T> result(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(static_cast<T>(0), static_cast<T>(1));
    for (size_t i = 0; i < size; ++i)
    {
        result[i] = dist(gen);
    }

    return result;
}


S3Token2Wav::S3Token2Wav(ggml_backend_t shared_backend)
{
    if (shared_backend) {
        backend = shared_backend;
        owns_backend = false;
    } else {
        owns_backend = true;
    }
}

S3Token2Wav::~S3Token2Wav()
{
    free_model();
}

bool S3Token2Wav::init_backend() {
    if (backend) {
        // Backend already initialized (shared or otherwise)
        return true;
    }

#ifdef GGML_USE_CUDA
    fprintf(stderr, "Trying CUDA backend...\n");
    backend = ggml_backend_cuda_init(0);  // Device 0
    if (backend) {
        fprintf(stderr, "CUDA backend initialized successfully\n");
        return true;
    }
    fprintf(stderr, "Failed to initialize CUDA backend, falling back...\n");
#endif

#ifdef GGML_USE_VULKAN
    fprintf(stderr, "Trying Vulkan backend...\n");
    backend = ggml_backend_vk_init(0);  // Device 0
    if (backend) {
        fprintf(stderr, "Vulkan backend initialized successfully\n");
        return true;
    }
    fprintf(stderr, "Failed to initialize Vulkan backend, falling back...\n");
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "Trying Metal backend...\n");
    backend = ggml_backend_metal_init();
    if (backend) {
        fprintf(stderr, "Metal backend initialized successfully\n");
        return true;
    }
    fprintf(stderr, "Failed to initialize Metal backend, falling back to CPU\n");
#endif

    fprintf(stderr, "Initializing CPU backend...\n");
    backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Failed to initialize CPU backend\n");
        return false;
    }

    fprintf(stderr, "CPU backend initialized successfully\n");
    return true;
}


bool S3Token2Wav::load_model(const std::string& model_path, bool is_meanflow)
{
    struct ggml_context* tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &tmp_ctx,
    };

    gguf_context* gguf_ctx = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "S3Token2Wav: gguf_init_from_file() failed\n");
        return false;
    }

    int num_tensors = static_cast<int>(gguf_get_n_tensors(gguf_ctx));
    printf("S3Token2Wav: num_tensors: %d\n", num_tensors);

    struct ggml_init_params params {
        /*.mem_size   = */ ggml_tensor_overhead()* (num_tensors + 5), // + 5 for noise tensor, istft, flow_decoder rand_noise
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ true,
    };
    if (!model_ctx)
    {
        model_ctx = ggml_init(params);
    }
    // Create single config for all components
    S3GenConfig cfg = is_meanflow ? S3GenConfig::meanflow_config() : S3GenConfig::default_config();

    // Create encoder, decoder, and flow — all share the same config
    auto encoder = std::make_unique<UpsampleConformerEncoder>(cfg);
    auto decoder = std::make_unique<ConditionalDecoder>(cfg);

    flow = std::make_unique<S3Token2Mel>(std::move(encoder), std::move(decoder), cfg);

    mel2wav = std::make_unique<S3GenModelData>();
    mel2wav->ctx = model_ctx;
    mel2wav->backend = backend;
    mel2wav->buffer = buffer;

    for (int i = 0; i < num_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor* dst = ggml_dup_tensor(model_ctx, src);
        ggml_set_name(dst, name);
    }
    S3GenDataLoader::create_gaussian_noise_tensor(*mel2wav);
    S3GenDataLoader::create_istft_context(*mel2wav);

    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);

    // Copy tensors from main memory to backend
    for (struct ggml_tensor* cur = ggml_get_first_tensor(model_ctx); cur != nullptr;
        cur = ggml_get_next_tensor(model_ctx, cur)) {
        struct ggml_tensor* src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        if (src)
        {
            size_t n_size = ggml_nbytes(src);
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
        }
    }

    if (!flow->load_all(model_ctx)) {
        return false;
    }

    printf("=== S3Gen Mel2Wav ===\n\n");
    // Organize F0 predictor tensors
    S3GenDataLoader::organize_f0_predictor(*mel2wav);

    // Organize source module tensors (noise tensor already created and allocated)
    S3GenDataLoader::organize_source_module(*mel2wav);

    // Fill noise tensor with Gaussian random values (after backend allocation)
    S3GenDataLoader::fill_gaussian_noise_tensor(*mel2wav);
    S3GenDataLoader::fill_istft_context(*mel2wav);

    // organize decode module
    S3GenDataLoader::organize_decode_module(*mel2wav);

    return true;
}

void S3Token2Wav::free_graph()
{
    if (allocr) {
        ggml_gallocr_free(allocr);
        allocr = nullptr;
    }
    if (graph_ctx) {
        ggml_free(graph_ctx);
        graph_ctx = nullptr;
    }
    if (flow) {
        flow->clear_rand_noise();
    }
    graph = nullptr;
}

void S3Token2Wav::free_model()
{
    free_graph();

    if (model_ctx) {
        ggml_free(model_ctx);
        model_ctx = nullptr;
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (backend) {
        if (owns_backend) {
            ggml_backend_free(backend);
        }
        backend = nullptr;
    }
    // unique_ptr members are automatically cleaned up, but reset explicitly for clarity
    flow.reset();
    mel2wav.reset();
}

bool S3Token2Wav::build_graph(ggml_context* ctx, ggml_tensor* token,
    ggml_tensor* prompt_token, ggml_tensor* prompt_feat, ggml_tensor* embedding,
    ggml_tensor* attention_mask, bool meanflow)
{
    graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 20, false);

    auto [mels, _]= flow->inference(ctx, token, prompt_token, prompt_feat, embedding, attention_mask);

    auto result = decode(ctx, mels, *mel2wav);

    ggml_set_name(mels, "output_mels");
    ggml_set_output(mels);
    ggml_set_name(result , "result_wav");
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    // ggml_build_forward_expand(graph, mels);

    if (!graph)
    {
        fprintf(stderr, "Failed to create graph");
        return false;
    }

    // Free previous allocator if exists (important for multiple inferences / chunking)
    if (allocr) {
        ggml_gallocr_free(allocr);
        allocr = nullptr;
    }

    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocr, graph))
    {
        fprintf(stderr, "Error in allocating graph");
        ggml_gallocr_free(allocr);
        allocr = nullptr;
        return false;
    }

    return true;
}

bool S3Token2Wav::compute_graph()
{
    printf("S3Token2Wav::compute_graph() called\n");
    printf("  backend = %p\n", (void*)backend);
    printf("  graph = %p\n", (void*)graph);

    auto start = std::chrono::high_resolution_clock::now();

    printf("  Calling ggml_backend_graph_compute...\n");
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "T3Inference: ggml_backend_graph_compute() failed\n");
        return false;
    }
    printf("  ggml_backend_graph_compute completed\n");

    auto end = std::chrono::high_resolution_clock::now();
    auto graph_exec_time = std::chrono::duration<double, std::milli>(
        end - start
    );
    printf("S3Gen Graph Exec Time: %f milliseconds\n", graph_exec_time.count());
    return true;
}

// === High-level generate implementation ===

// Audio constants for trimming
static constexpr int SAMPLE_RATE = 24000;
static constexpr int SAMPLES_PER_TOKEN = SAMPLE_RATE / 25;  // 960 samples per speech token

static size_t calc_size_from_shape(const int32_t* shape, size_t n_dims) {
    size_t size = 1;
    for (size_t i = 0; i < n_dims; i++) {
        size *= shape[i];
    }
    return size;
}

bool S3Token2Wav::generate(
    const std::vector<int>& speech_tokens,
    const GenerateParams& gen_params,
    float** output_audio,
    size_t* audio_size)
{
    // Validate inputs
    if (!gen_params.prompt_tokens || !gen_params.prompt_tokens_shape || gen_params.prompt_tokens_n_dims == 0) {
        fprintf(stderr, "Missing prompt_tokens buffer or shape\n");
        return false;
    }
    if (!gen_params.prompt_feat || !gen_params.prompt_feat_shape || gen_params.prompt_feat_n_dims == 0) {
        fprintf(stderr, "Missing prompt_feat buffer or shape\n");
        return false;
    }
    if (!gen_params.speaker_emb || !gen_params.speaker_emb_shape || gen_params.speaker_emb_n_dims == 0) {
        fprintf(stderr, "Missing speaker_emb buffer or shape\n");
        return false;
    }

    // Calculate sizes from shapes
    size_t prompt_tokens_size = calc_size_from_shape(gen_params.prompt_tokens_shape, gen_params.prompt_tokens_n_dims);
    size_t prompt_feat_size = calc_size_from_shape(gen_params.prompt_feat_shape, gen_params.prompt_feat_n_dims);
    size_t speaker_emb_size = calc_size_from_shape(gen_params.speaker_emb_shape, gen_params.speaker_emb_n_dims);

    // Extract dimensions
    size_t token_seq_len = speech_tokens.size();
    size_t prompt_token_seq_len = (gen_params.prompt_tokens_n_dims == 2)
        ? gen_params.prompt_tokens_shape[1]
        : prompt_tokens_size;

    size_t prompt_feat_frames, prompt_feat_dim;
    if (gen_params.prompt_feat_n_dims == 3) {
        prompt_feat_frames = gen_params.prompt_feat_shape[2];
        prompt_feat_dim = gen_params.prompt_feat_shape[1];
    } else if (gen_params.prompt_feat_n_dims == 2) {
        prompt_feat_frames = gen_params.prompt_feat_shape[1];
        prompt_feat_dim = gen_params.prompt_feat_shape[0];
    } else {
        prompt_feat_dim = 80;
        prompt_feat_frames = prompt_feat_size / prompt_feat_dim;
    }

    size_t speaker_emb_dim = (gen_params.speaker_emb_n_dims == 2)
        ? gen_params.speaker_emb_shape[1]
        : speaker_emb_size;

    // Free any previous graph
    free_graph();

    // Create graph context
    size_t compute_mem_size = 512 * 1024 * 1024;  // 512 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead() + compute_mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    graph_ctx = ggml_init(params);
    if (!graph_ctx) {
        fprintf(stderr, "S3Gen: Failed to create graph context\n");
        return false;
    }

    // Create input tensors
    ggml_tensor* token = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32, token_seq_len, 1);
    ggml_set_name(token, "token");
    ggml_set_input(token);

    ggml_tensor* prompt_token = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32, prompt_token_seq_len, 1);
    ggml_set_name(prompt_token, "prompt_token");
    ggml_set_input(prompt_token);

    ggml_tensor* prompt_feat = ggml_new_tensor_3d(graph_ctx, GGML_TYPE_F32,
        prompt_feat_dim, prompt_feat_frames, 1);
    ggml_set_name(prompt_feat, "prompt_feat");
    ggml_set_input(prompt_feat);

    ggml_tensor* embedding = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, speaker_emb_dim, 1);
    ggml_set_name(embedding, "embedding");
    ggml_set_input(embedding);

    size_t total_seq_len = prompt_token_seq_len + token_seq_len;
    ggml_tensor* attention_mask = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, total_seq_len, 1);
    ggml_set_name(attention_mask, "attention_mask");
    ggml_set_input(attention_mask);

    ggml_tensor* rand_noise = ggml_new_tensor_3d(graph_ctx, GGML_TYPE_F32, 50 * 300, 80, 1);
    ggml_set_name(rand_noise, "rand_noise");
    ggml_set_input(rand_noise);

    flow->set_rand_noise(rand_noise);

    // Build graph
    if (!build_graph(graph_ctx, token, prompt_token, prompt_feat,
                     embedding, attention_mask, flow->config.meanflow)) {
        fprintf(stderr, "S3Gen: Failed to build graph\n");
        free_graph();
        return false;
    }

    // Convert prompt tokens from int64 to int32
    std::vector<int32_t> prompt_tokens_i32(prompt_tokens_size);
    for (size_t i = 0; i < prompt_tokens_size; i++) {
        prompt_tokens_i32[i] = static_cast<int32_t>(gen_params.prompt_tokens[i]);
    }

    // Convert speech tokens to int32
    std::vector<int32_t> speech_tokens_i32(token_seq_len);
    for (size_t i = 0; i < token_seq_len; i++) {
        speech_tokens_i32[i] = static_cast<int32_t>(speech_tokens[i]);
    }

    // Set input tensor data
    ggml_backend_tensor_set(token, speech_tokens_i32.data(), 0,
                            token_seq_len * sizeof(int32_t));
    ggml_backend_tensor_set(prompt_token, prompt_tokens_i32.data(), 0,
                            prompt_token_seq_len * sizeof(int32_t));
    ggml_backend_tensor_set(prompt_feat, gen_params.prompt_feat, 0,
                            prompt_feat_size * sizeof(float));
    ggml_backend_tensor_set(embedding, gen_params.speaker_emb, 0,
                            speaker_emb_size * sizeof(float));

    // Attention mask: 1.0 for all valid positions
    std::vector<float> attention_mask_data(total_seq_len, 1.0f);
    ggml_backend_tensor_set(attention_mask, attention_mask_data.data(), 0,
                            total_seq_len * sizeof(float));

    // Random noise for flow matching
    std::vector<float> rand_noise_data = randn_like<float>(50 * 300 * 80);
    ggml_backend_tensor_set(rand_noise, rand_noise_data.data(), 0,
                            rand_noise_data.size() * sizeof(float));

    // Compute
    if (!compute_graph()) {
        fprintf(stderr, "Failed to compute graph\n");
        free_graph();
        return false;
    }

    // Extract output
    ggml_tensor* result_wav = ggml_graph_get_tensor(graph, "result_wav");
    if (!result_wav) {
        fprintf(stderr, "Output tensor 'result_wav' not found\n");
        free_graph();
        return false;
    }

    size_t total_elements = ggml_nelements(result_wav);
    *audio_size = total_elements;
    auto audio_buffer = std::make_unique<float[]>(total_elements);
    ggml_backend_tensor_get(result_wav, audio_buffer.get(), 0, total_elements * sizeof(float));
    *output_audio = audio_buffer.release();

    return true;
}
