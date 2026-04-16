#include "overlap-add.cuh"

// CUDA kernel for overlap-add operation with proper window normalization
// Each thread computes one output sample by accumulating windowed frame contributions
static __global__ void overlap_add_f32_kernel(
    const float * __restrict__ frames,      // [num_frames, win_length, batch] unwindowed
    const float * __restrict__ window,      // [win_length] window function
    float * __restrict__ output,            // [output_length, batch]
    const int64_t num_frames,
    const int64_t win_length,
    const int64_t batch_size,
    const int64_t output_length,
    const int hop_length,
    const int center_padding
) {
    const int64_t out_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t batch_idx = blockIdx.y;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    const int64_t padded_pos = out_idx + center_padding;

    float signal_sum = 0.0f;
    float window_sum = 0.0f;

    const int64_t first_frame = max((int64_t)0,
        (padded_pos - center_padding - win_length + 1 + hop_length - 1) / hop_length);
    const int64_t last_frame  = min(num_frames - 1,
        (padded_pos - center_padding) / hop_length);

    for (int64_t frame_idx = first_frame; frame_idx <= last_frame; frame_idx++) {
        const int64_t frame_start      = frame_idx * hop_length + center_padding;
        const int64_t offset_in_frame  = padded_pos - frame_start;

        if (offset_in_frame >= 0 && offset_in_frame < win_length) {
            // GGML layout for [num_frames, win_length, batch]:
            // element [frame, sample, batch] = frame + sample*num_frames + batch*num_frames*win_length
            const int64_t frame_element_idx =
                frame_idx +
                offset_in_frame * num_frames +
                batch_idx       * num_frames * win_length;

            float window_val  = window[offset_in_frame];
            signal_sum       += frames[frame_element_idx] * window_val;
            window_sum       += window_val;
        }
    }

    const float eps   = 1e-11f;
    float result      = (window_sum > eps) ? (signal_sum / window_sum) : signal_sum;
    output[batch_idx * output_length + out_idx] = result;
}

// Optimized kernel using shared memory for window
static __global__ void overlap_add_f32_kernel_shared(
    const float * __restrict__ frames,
    const float * __restrict__ window,
    float * __restrict__ output,
    const int64_t num_frames,
    const int64_t win_length,
    const int64_t batch_size,
    const int64_t output_length,
    const int hop_length,
    const int center_padding
) {
    extern __shared__ float shared_window[];

    const int64_t out_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t batch_idx = blockIdx.y;

    for (int i = threadIdx.x; i < win_length; i += blockDim.x) {
        shared_window[i] = window[i];
    }
    __syncthreads();

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    const int64_t padded_pos  = out_idx + center_padding;
    const int64_t first_frame = max((int64_t)0,
        (padded_pos - center_padding - win_length + 1 + hop_length - 1) / hop_length);
    const int64_t last_frame  = min(num_frames - 1,
        (padded_pos - center_padding) / hop_length);

    float signal_sum = 0.0f;
    float window_sum = 0.0f;

    for (int64_t frame_idx = first_frame; frame_idx <= last_frame; frame_idx++) {
        const int64_t frame_start     = frame_idx * hop_length + center_padding;
        const int64_t offset_in_frame = padded_pos - frame_start;

        if (offset_in_frame >= 0 && offset_in_frame < win_length) {
            const int64_t frame_element_idx =
                frame_idx +
                offset_in_frame * num_frames +
                batch_idx       * num_frames * win_length;

            float window_val  = shared_window[offset_in_frame];
            signal_sum       += frames[frame_element_idx] * window_val;
            window_sum       += window_val;
        }
    }

    const float eps = 1e-11f;
    float result    = (window_sum > eps) ? (signal_sum / window_sum) : signal_sum;
    output[batch_idx * output_length + out_idx] = result;
}

static void overlap_add_f32_cuda(
    const float * frames,
    const float * window,
    float       * output,
    const int64_t num_frames,
    const int64_t win_length,
    const int64_t batch_size,
    const int64_t output_length,
    const int     hop_length,
    const int     center_padding,
    cudaStream_t  stream
) {
    const int block_size = CUDA_OVERLAP_ADD_BLOCK_SIZE;

    if (win_length <= 1024) {
        const size_t shared_mem_size = win_length * sizeof(float);
        dim3 blocks((output_length + block_size - 1) / block_size, batch_size);
        overlap_add_f32_kernel_shared<<<blocks, block_size, shared_mem_size, stream>>>(
            frames, window, output,
            num_frames, win_length, batch_size, output_length,
            hop_length, center_padding);
    } else {
        dim3 blocks((output_length + block_size - 1) / block_size, batch_size);
        overlap_add_f32_kernel<<<blocks, block_size, 0, stream>>>(
            frames, window, output,
            num_frames, win_length, batch_size, output_length,
            hop_length, center_padding);
    }
}

void ggml_cuda_op_overlap_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // time_frames
    const ggml_tensor * src1 = dst->src[1];  // window

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t * opts          = (const int32_t *) dst->op_params;
    const int       hop_length    = opts[0];
    const int       center_padding = opts[1];

    const int64_t num_frames   = src0->ne[0];
    const int64_t win_length   = src0->ne[1];
    const int64_t batch_size   = src0->ne[2];
    const int64_t output_length = dst->ne[0];

    GGML_ASSERT(src1->ne[0] == win_length);

    overlap_add_f32_cuda(
        (const float *) src0->data,
        (const float *) src1->data,
        (float *)       dst->data,
        num_frames, win_length, batch_size, output_length,
        hop_length, center_padding,
        ctx.stream());
}
