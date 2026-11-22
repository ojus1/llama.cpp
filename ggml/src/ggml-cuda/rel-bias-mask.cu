#include "rel-bias-mask.cuh"
#include "convert.cuh"

// res[j, t, h, s] = (dst_t) (table[h, rel_pos[j, s*T + t], s*T + t] + (float) mask[j, t, 0, s])
// one thread per kv cell j and row (t, s); the thread loops over the heads, so the bucket and the mask value
// are loaded once per cell and the writes of a warp stay contiguous for each head
// all offsets are 64-bit: the output can exceed 2^32 elements (e.g. 131072 cells x 2048 tokens x 24 heads)
template <typename mask_t, typename dst_t>
static __global__ void k_rel_bias_mask(
        const char * __restrict__ table, const char * __restrict__ rel_pos, const char * __restrict__ mask, char * __restrict__ dst,
        const int64_t n_kv, const int64_t T, const int64_t n_rows, const int64_t n_head, const int64_t n_rel,
        const int64_t tb0, const int64_t tb1, const int64_t tb2,
        const int64_t rb1,
        const int64_t mb1, const int64_t mb3,
        const int64_t db1, const int64_t db2, const int64_t db3) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_kv) {
        return;
    }

    for (int64_t r = blockIdx.y; r < n_rows; r += gridDim.y) {
        const int64_t t   = r % T;
        const int64_t s   = r / T;
        const int64_t tok = r; // == s*T + t

        int32_t idx = ((const int32_t *) (rel_pos + tok*rb1))[j];
        idx = idx < 0 ? 0 : (idx >= n_rel ? (int32_t) (n_rel - 1) : idx);

        const float m = ggml_cuda_cast<float>(((const mask_t *) (mask + t*mb1 + s*mb3))[j]);

        const char * trow = table + tok*tb2 + idx*tb1;
        char       * drow = dst + t*db1 + s*db3 + j*(int64_t) sizeof(dst_t);

#pragma unroll 4
        for (int64_t h = 0; h < n_head; ++h) {
            const float b = *(const float *) (trow + h*tb0);
            *(dst_t *) (drow + h*db2) = ggml_cuda_cast<dst_t>(b + m);
        }
    }
}

template <typename mask_t, typename dst_t>
static void rel_bias_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * table   = dst->src[0];
    const ggml_tensor * rel_pos = dst->src[1];
    const ggml_tensor * mask    = dst->src[2];

    const int64_t n_kv   = dst->ne[0];
    const int64_t T      = dst->ne[1];
    const int64_t n_head = dst->ne[2];
    const int64_t n_rows = T*dst->ne[3];
    const int64_t n_rel  = table->ne[1];

    const int block = 256;
    const int64_t gx = (n_kv + block - 1)/block;
    GGML_ASSERT(gx <= INT_MAX);
    const dim3 grid((unsigned) gx, (unsigned) std::min<int64_t>(n_rows, 65535), 1);

    k_rel_bias_mask<mask_t, dst_t><<<grid, block, 0, ctx.stream()>>>(
        (const char *) table->data, (const char *) rel_pos->data, (const char *) mask->data, (char *) dst->data,
        n_kv, T, n_rows, n_head, n_rel,
        table->nb[0], table->nb[1], table->nb[2],
        rel_pos->nb[1],
        mask->nb[1], mask->nb[3],
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_rel_bias_mask_supported(const ggml_tensor * op) {
    const ggml_tensor * table   = op->src[0];
    const ggml_tensor * rel_pos = op->src[1];
    const ggml_tensor * mask    = op->src[2];
    return table->type == GGML_TYPE_F32 && rel_pos->type == GGML_TYPE_I32 &&
        (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) &&
        (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
        rel_pos->nb[0] == sizeof(int32_t) && mask->nb[0] == ggml_type_size(mask->type) &&
        op->nb[0] == ggml_type_size(op->type);
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask = dst->src[2];
    if (mask->type == GGML_TYPE_F16) {
        if (dst->type == GGML_TYPE_F16) {
            rel_bias_mask_cuda<half, half>(ctx, dst);
        } else {
            rel_bias_mask_cuda<half, float>(ctx, dst);
        }
    } else {
        if (dst->type == GGML_TYPE_F16) {
            rel_bias_mask_cuda<float, half>(ctx, dst);
        } else {
            rel_bias_mask_cuda<float, float>(ctx, dst);
        }
    }
}
