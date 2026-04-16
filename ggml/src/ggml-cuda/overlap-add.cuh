#include "common.cuh"

#define CUDA_OVERLAP_ADD_BLOCK_SIZE 256

void ggml_cuda_op_overlap_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
