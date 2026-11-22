#include "common.cuh"

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_rel_bias_mask_supported(const ggml_tensor * op);
