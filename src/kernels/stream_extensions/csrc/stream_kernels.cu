#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <type_traits>


#define DISPATCH_TYPES(TYPE1, TYPE2, NAME, ...)                                \
    switch (TYPE1) {                                                           \
        case at::ScalarType::Double: {                                         \
            using scalar_t_0 = double;                                         \
            switch (TYPE2) {                                                   \
                case at::ScalarType::Double: {                                 \
                    using scalar_t_1 = double;                                 \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Float: {                                  \
                    using scalar_t_1 = float;                                  \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::BFloat16: {                               \
                    using scalar_t_1 = nv_bfloat16;                            \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Half: {                                   \
                    using scalar_t_1 = nv_half;                                \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                default:                                                       \
                    TORCH_CHECK(false,                                         \
                                #NAME,                                         \
                                " not supported for '",                        \
                                toString(TYPE1),                               \
                                "' with '",                                    \
                                toString(TYPE2),                               \
                                "'");                                          \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case at::ScalarType::Float: {                                          \
            using scalar_t_0 = float;                                          \
            switch (TYPE2) {                                                   \
                case at::ScalarType::Double: {                                 \
                    using scalar_t_1 = double;                                 \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Float: {                                  \
                    using scalar_t_1 = float;                                  \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::BFloat16: {                               \
                    using scalar_t_1 = nv_bfloat16;                            \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Half: {                                   \
                    using scalar_t_1 = nv_half;                                \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                default:                                                       \
                    TORCH_CHECK(false,                                         \
                                #NAME,                                         \
                                " not supported for '",                        \
                                toString(TYPE1),                               \
                                "' with '",                                    \
                                toString(TYPE2),                               \
                                "'");                                          \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case at::ScalarType::BFloat16: {                                       \
            using scalar_t_0 = nv_bfloat16;                                    \
            switch (TYPE2) {                                                   \
                case at::ScalarType::Double: {                                 \
                    using scalar_t_1 = double;                                 \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Float: {                                  \
                    using scalar_t_1 = float;                                  \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::BFloat16: {                               \
                    using scalar_t_1 = nv_bfloat16;                            \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Half: {                                   \
                    using scalar_t_1 = nv_half;                                \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                default:                                                       \
                    TORCH_CHECK(false,                                         \
                                #NAME,                                         \
                                " not supported for '",                        \
                                toString(TYPE1),                               \
                                "' with '",                                    \
                                toString(TYPE2),                               \
                                "'");                                          \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case at::ScalarType::Half: {                                           \
            using scalar_t_0 = nv_half;                                        \
            switch (TYPE2) {                                                   \
                case at::ScalarType::Double: {                                 \
                    using scalar_t_1 = double;                                 \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Float: {                                  \
                    using scalar_t_1 = float;                                  \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::BFloat16: {                               \
                    using scalar_t_1 = nv_bfloat16;                            \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                case at::ScalarType::Half: {                                   \
                    using scalar_t_1 = nv_half;                                \
                    __VA_ARGS__;                                               \
                    break;                                                     \
                }                                                              \
                default:                                                       \
                    TORCH_CHECK(false,                                         \
                                #NAME,                                         \
                                " not supported for '",                        \
                                toString(TYPE1),                               \
                                "' with '",                                    \
                                toString(TYPE2),                               \
                                "'");                                          \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        default:                                                               \
            TORCH_CHECK(false,                                                 \
                        #NAME,                                                 \
                        " not supported for '",                                \
                        toString(TYPE1),                                       \
                        "' with '",                                            \
                        toString(TYPE2),                                       \
                        "'");                                                  \
    }

using cuHalfComplex     = nv_half2;
using cuBFloat16Complex = nv_bfloat162;

template<typename T>
__device__ double to_double(T src)
{
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        return src;
    }
    else if constexpr (std::is_same_v<T, nv_half>) {
        return __half2float(src);
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __bfloat162float(src);
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for to_double");
    }
}

template<typename T>
__device__ T from_double(double src)
{
    if constexpr (std::is_same_v<T, double>) {
        return src;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(src);
    }
    else if constexpr (std::is_same_v<T, nv_half>) {
        return __double2half(src);
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __double2bfloat16(src);
    }
    else {
        static_assert(!std::is_same_v<T, T>,
                      "Unsupported type for from_double");
    }
}

template<typename T>
__device__ float to_float(T src)
{
    if constexpr (std::is_same_v<T, float>) {
        return src;
    }
    else if constexpr (std::is_same_v<T, double>) {
        return static_cast<float>(src);
    }
    else if constexpr (std::is_same_v<T, nv_half>) {
        return __half2float(src);
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __bfloat162float(src);
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for to_float");
    }
}

template<typename T>
__device__ T from_float(float src)
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return src;
    }
    else if constexpr (std::is_same_v<T, nv_half>) {
        return __float2half(src);
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return __float2bfloat16(src);
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type for from_float");
    }
}

template<typename T>
T* get_ptr(torch::Tensor& src)
{
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        return src.data_ptr<T>();
    }
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return reinterpret_cast<T*>(src.data_ptr<c10::complex<double>>());
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return reinterpret_cast<T*>(src.data_ptr<c10::BFloat16>());
    }
    else if constexpr (std::is_same_v<T, nv_half>) {
        return reinterpret_cast<T*>(src.data_ptr<c10::Half>());
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        return reinterpret_cast<T*>(src.data_ptr<int64_t>());
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type");
    }
}

template<typename T>
const T* get_ptr(const torch::Tensor& src)
{
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        return src.data_ptr<T>();
    }
    else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        return reinterpret_cast<const T*>(src.data_ptr<c10::complex<double>>());
    }
    else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return reinterpret_cast<const T*>(src.data_ptr<c10::BFloat16>());
    }
    else if constexpr (std::is_same_v<T, nv_half>) {
        return reinterpret_cast<const T*>(src.data_ptr<c10::Half>());
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        return reinterpret_cast<const T*>(src.data_ptr<int64_t>());
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported type");
    }
}

template<typename scalar_t_0, typename scalar_t_1>
__global__ void fused_rope_qk_forward(const int64_t b,
                                      const int64_t total_seq_len,
                                      const int64_t h,
                                      const int64_t d,
                                      const int64_t stride_b,
                                      const int64_t stride_s,
                                      const int64_t stride_h,
                                      const int64_t stride_d,
                                      const int64_t o_stride_b,
                                      const int64_t o_stride_s,
                                      const int64_t o_stride_h,
                                      const int64_t o_stride_d,
                                      const scalar_t_0* __restrict__ q_src,
                                      const scalar_t_0* __restrict__ k_src,
                                      const cuDoubleComplex* __restrict__ freq,
                                      const int64_t* __restrict__ grid,
                                      scalar_t_1* __restrict__ q_dst,
                                      scalar_t_1* __restrict__ k_dst)
{
    if (blockIdx.x >= total_seq_len || blockIdx.y >= b) {
        return;
    }

    int64_t ppf             = grid[blockIdx.y * 3];
    int64_t pph             = grid[blockIdx.y * 3 + 1];
    int64_t ppw             = grid[blockIdx.y * 3 + 2];
    int64_t total_video_len = ppf * pph * ppw;

    int64_t f_split_id = d / 2 - 2 * (d / 6);
    int64_t h_split_id = d / 6;
    int64_t half_d     = d / 2;
    int64_t fid        = blockIdx.x / (pph * ppw);
    int64_t hid        = (blockIdx.x % (pph * ppw)) / ppw;
    int64_t wid        = blockIdx.x % ppw;

    extern __shared__ cuDoubleComplex freq_shared[];
    for (int32_t d_id = threadIdx.x; d_id < half_d; d_id += blockDim.x) {
        if (blockIdx.x < total_video_len) {
            cuDoubleComplex temp;
            if (d_id < f_split_id) {
                temp = freq[fid * half_d + d_id];
            }
            else if (d_id < f_split_id + h_split_id) {
                temp = freq[hid * half_d + d_id];
            }
            else {
                temp = freq[wid * half_d + d_id];
            }
            freq_shared[d_id] = temp;
        }
    }
    __syncthreads();

#pragma unroll
    for (int32_t d_id = threadIdx.x; d_id < half_d; d_id += blockDim.x) {
        cuDoubleComplex freq_value;

        if (blockIdx.x < total_video_len) {
            freq_value = freq_shared[d_id];
        }

        int32_t base_offset_src =
            blockIdx.y * stride_b + blockIdx.x * stride_s + (d_id * 2 + 0) * stride_d;
        int32_t base_offset_dst =
            blockIdx.y * o_stride_b + blockIdx.x * o_stride_s + (d_id * 2 + 0) * o_stride_d;

#pragma unroll
        for (int32_t h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
            int32_t offset_src = base_offset_src + h_id * stride_h;
            int32_t offset_dst = base_offset_dst + h_id * o_stride_h;
            if (blockIdx.x < total_video_len) {
                double q_src_real = to_double(q_src[offset_src]);
                double q_src_imag = to_double(q_src[offset_src + stride_d]);

                double k_src_real = to_double(k_src[offset_src]);
                double k_src_imag = to_double(k_src[offset_src + stride_d]);

                cuDoubleComplex q_src_value =
                    make_cuDoubleComplex(q_src_real, q_src_imag);
                cuDoubleComplex k_src_value =
                    make_cuDoubleComplex(k_src_real, k_src_imag);

                cuDoubleComplex q_dst_value = cuCmul(q_src_value, freq_value);
                cuDoubleComplex k_dst_value = cuCmul(k_src_value, freq_value);

                q_dst[offset_dst] =
                    from_double<scalar_t_1>(cuCreal(q_dst_value));
                q_dst[offset_dst + o_stride_d] =
                    from_double<scalar_t_1>(cuCimag(q_dst_value));

                k_dst[offset_dst] =
                    from_double<scalar_t_1>(cuCreal(k_dst_value));
                k_dst[offset_dst + o_stride_d] =
                    from_double<scalar_t_1>(cuCimag(k_dst_value));
            }
            else {
                q_dst[offset_dst] =
                    from_double<scalar_t_1>(to_double(q_src[offset_src]));
                q_dst[offset_dst + o_stride_d] = from_double<scalar_t_1>(
                    to_double(q_src[offset_src + stride_d]));

                k_dst[offset_dst] =
                    from_double<scalar_t_1>(to_double(k_src[offset_src]));
                k_dst[offset_dst + o_stride_d] = from_double<scalar_t_1>(
                    to_double(k_src[offset_src + stride_d]));
            }
        }
    }
}

template<typename scalar_t_0, typename scalar_t_1>
void fused_rope_qk_forward_call(const int64_t b,
                                const int64_t total_seq_len,
                                const int64_t h,
                                const int64_t d,
                                const int64_t stride_b,
                                const int64_t stride_s,
                                const int64_t stride_h,
                                const int64_t stride_d,
                                const int64_t o_stride_b,
                                const int64_t o_stride_s,
                                const int64_t o_stride_h,
                                const int64_t o_stride_d,
                                const scalar_t_0* __restrict__ q_src,
                                const scalar_t_0* __restrict__ k_src,
                                const cuDoubleComplex* __restrict__ freq,
                                const int64_t* __restrict__ grid,
                                scalar_t_1* __restrict__ q_dst,
                                scalar_t_1* __restrict__ k_dst)
{
    auto stream = c10::cuda::getCurrentCUDAStream();

    dim3 grid_dims(total_seq_len, b);
    dim3 block_dims(C10_WARP_SIZE, h < 16 ? 4 : 8);

    size_t shared_mem_size = (d / 2) * sizeof(cuDoubleComplex);

    fused_rope_qk_forward<<<grid_dims, block_dims, shared_mem_size, stream>>>(
        b,
        total_seq_len,
        h,
        d,
        stride_b,
        stride_s,
        stride_h,
        stride_d,
        o_stride_b,
        o_stride_s,
        o_stride_h,
        o_stride_d,
        q_src,
        k_src,
        freq,
        grid,
        q_dst,
        k_dst);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_rope_qk_fwd(const torch::Tensor& q_src,
                       const torch::Tensor& k_src,
                       const torch::Tensor& freq,
                       const torch::Tensor& grid,
                       torch::Tensor&       q_dst,
                       torch::Tensor&       k_dst)
{
    c10::ScalarType src_type = q_src.scalar_type();
    c10::ScalarType dst_type = q_dst.scalar_type();
    TORCH_CHECK(
        src_type == c10::ScalarType::Double
            || src_type == c10::ScalarType::Float
            || src_type == c10::ScalarType::BFloat16
            || src_type == c10::ScalarType::Half,
        "expected the dtype of the src tensor is torch.float64 or torch.float32 or torch.bfloat16 or torch.float16");
    TORCH_CHECK(
        dst_type == c10::ScalarType::Double
            || dst_type == c10::ScalarType::Float
            || dst_type == c10::ScalarType::BFloat16
            || dst_type == c10::ScalarType::Half,
        "expected the dtype of the dst tensor is torch.float64 or torch.float32 or torch.bfloat16 or torch.float16");
    TORCH_CHECK(
        src_type == k_src.scalar_type(),
        "expected the dtype of the k_src tensor is the same as q_src tensor");
    TORCH_CHECK(
        dst_type == k_dst.scalar_type(),
        "expected the dtype of the k_dst tensor is the same as q_dst tensor");
    TORCH_CHECK(freq.is_cuda(), "expected freq is_cuda");
    TORCH_CHECK(freq.scalar_type() == c10::ScalarType::ComplexDouble,
                "expected the dtype of the freq tensor is torch.complex128");
    TORCH_CHECK(grid.is_cuda(), "expected grid is_cuda");
    TORCH_CHECK(grid.scalar_type() == c10::ScalarType::Long,
                "expected the dtype of the grid tensor is torch.int64");
    TORCH_CHECK(grid.size(1) == 3, "expected the grid.size(1) == 3");

    TORCH_CHECK(
        q_src.sizes() == k_src.sizes(),
        "expected the shape of the k_src tensor is the same as q_src tensor");
    TORCH_CHECK(
        q_dst.sizes() == k_dst.sizes(),
        "expected the shape of the k_dst tensor is the same as q_dst tensor");
    TORCH_CHECK(
        q_src.strides() == k_src.strides(),
        "expected the strides of the k_src tensor is the same as q_src tensor");
    TORCH_CHECK(
        q_dst.strides() == k_dst.strides(),
        "expected the strides of the k_dst tensor is the same as q_dst tensor");

    const int64_t b             = q_src.size(0);
    const int64_t total_seq_len = q_src.size(1);
    const int64_t h             = q_src.size(2);
    const int64_t d             = q_src.size(3);

    const int64_t stride_b = q_src.stride(0);
    const int64_t stride_s = q_src.stride(1);
    const int64_t stride_h = q_src.stride(2);
    const int64_t stride_d = q_src.stride(3);

    const int64_t o_stride_b = q_dst.stride(0);
    const int64_t o_stride_s = q_dst.stride(1);
    const int64_t o_stride_h = q_dst.stride(2);
    const int64_t o_stride_d = q_dst.stride(3);

    DISPATCH_TYPES(src_type,
                   dst_type,
                   "forward_kernel_qk_partial_call",
                   fused_rope_qk_forward_call(b,
                                              total_seq_len,
                                              h,
                                              d,
                                              stride_b,
                                              stride_s,
                                              stride_h,
                                              stride_d,
                                              o_stride_b,
                                              o_stride_s,
                                              o_stride_h,
                                              o_stride_d,
                                              get_ptr<scalar_t_0>(q_src),
                                              get_ptr<scalar_t_0>(k_src),
                                              get_ptr<cuDoubleComplex>(freq),
                                              get_ptr<int64_t>(grid),
                                              get_ptr<scalar_t_1>(q_dst),
                                              get_ptr<scalar_t_1>(k_dst)););
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_rope_qk_fwd",
          &fused_rope_qk_fwd,
          "Fused Rotary Positional QK Embedding -- Forward.",
          py::arg("q_src"),
          py::arg("k_src"),
          py::arg("freq"),
          py::arg("grid"),
          py::arg("q_dst"),
          py::arg("k_dst"),
          py::call_guard<py::gil_scoped_release>());
}