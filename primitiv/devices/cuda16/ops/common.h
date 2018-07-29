#ifndef PRIMITIV_DEVICES_CUDA16_OPS_COMMON_H_
#define PRIMITIV_DEVICES_CUDA16_OPS_COMMON_H_

#include <cuda_fp16.h>

/*
 * Macros for device functions.
 */

#define IDX (threadIdx.x + blockIdx.x * blockDim.x)
#define IDY (threadIdx.y + blockIdx.y * blockDim.y)

#define FLOAT_POSITIVE_INFINITY ::__uint_as_float(0x7f800000)
#define FLOAT_NEGATIVE_INFINITY ::__uint_as_float(0xff800000)

#define X_VAL ::__half2float(px[i])
#define Y_VAL ::__half2float(py[i])
#define GX_VAL ::__half2float(pgx[i])
#define GY_VAL ::__half2float(pgy[i])

#define INPLACE_ADD(addr, val) { \
  *(addr) = ::__float2half(::__half2float(*(addr)) + (val)); \
}

#define DECL_ATOMIC_OP(name, op) \
__device__ void name(half *base, std::size_t offset, float value) { \
  const std::uint32_t shift = 16 * (offset & 1); \
  const std::uint32_t filter = 0xffff << (16 - shift); \
  std::uint32_t * const addr \
    = reinterpret_cast<std::uint32_t *>(base) + (offset >> 1); \
  std::uint32_t oldval = *addr; \
  std::uint32_t assumed; \
  do { \
    assumed = oldval; \
    const half a = ::__ushort_as_half((oldval >> shift) & 0xffff); \
    const half b = ::__float2half(op(::__half2float(a), value)); \
    const std::uint32_t newval \
      = (oldval & filter) | (::__half_as_ushort(b) << shift); \
    oldval = ::atomicCAS(addr, assumed, newval); \
  } while (oldval != assumed); \
}

#define CUDA16_KERNEL_FW_X(name, op) \
__global__ void name##_fw_dev(const half *px, std::uint32_t size, half *py) { \
  const std::uint32_t i = IDX; \
  if (i < size) py[i] = ::__float2half(op); \
}

#define CUDA16_KERNEL_BW_X(name, op) \
__global__ void name##_bw_dev( \
    const half *px, const half *py, const half *pgy, std::uint32_t size, \
    half *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const std::uint32_t i = IDX; \
  if (i < size) INPLACE_ADD(pgx + i, (op)); \
}

#define CUDA16_KERNEL_FW_X_CONST(name, op) \
__global__ void name##_fw_dev( \
    const half *px, float k, std::uint32_t size, half *py) { \
  const std::uint32_t i = IDX; \
  if (i < size) py[i] = ::__float2half(op); \
}

#define CUDA16_KERNEL_BW_X_CONST(name, op) \
__global__ void name##_bw_dev( \
    const half *px, const half *py, const half *pgy, float k, \
    std::uint32_t size, half *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const std::uint32_t i = IDX; \
  if (i < size) INPLACE_ADD(pgx + i, (op)); \
}

#define CUDA16_KERNEL_FW_X_SCALAR_R(name, op) \
__global__ void name##_fw_dev( \
    const half *px, const half *pk, std::uint32_t size, std::uint32_t mbx, \
    std::uint32_t mbk, half *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) { \
    py[i + shift] = ::__float2half(op( \
          ::__half2float(px[i + mbx * shift]), \
          ::__half2float(pk[mbk * blockIdx.y]))); \
  } \
}

#define CUDA16_KERNEL_FW_X_SCALAR_L(name, op) \
__global__ void name##_fw_dev( \
    const half *px, const half *pk, std::uint32_t size, std::uint32_t mbx, \
    std::uint32_t mbk, half *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) { \
    py[i + shift] = ::__float2half(op( \
          ::__half2float(pk[mbk * blockIdx.y]), \
          ::__half2float(px[i + mbx * shift]))); \
  } \
}

#define CUDA16_KERNEL_FW_AB(name, op) \
__global__ void name##_fw_dev( \
    const half *pa, const half *pb, std::uint32_t size, std::uint32_t mba, \
    std::uint32_t mbb, half *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) { \
    py[i + shift] = ::__float2half(op( \
          ::__half2float(pa[i + mba * shift]), \
          ::__half2float(pb[i + mbb * shift]))); \
  } \
}

#define CUDA16DEV_KERNEL_FW_X(name, op)
#define CUDA16DEV_KERNEL_BW_X(name, op)
#define CUDA16DEV_KERNEL_FW_X_CONST(name, op)
#define CUDA16DEV_KERNEL_BW_X_CONST(name, op)
#define CUDA16DEV_KERNEL_FW_X_SCALAR_R(name, op)
#define CUDA16DEV_KERNEL_FW_X_SCALAR_L(name, op)
#define CUDA16DEV_KERNEL_FW_AB(name, op)

/*
 * Macros for primitiv::Device overrides.
 */

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define CDATA(type, x) static_cast<const type *>(get_handle(x))
#define MDATA(type, x) static_cast<type *>(get_mutable_handle(x))

#define CUDA16_DEV_FW_X(name) \
void CUDA16::name##_fw_impl(const Tensor &x, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(half, x), size, MDATA(half, y)); \
}

#define CUDA16_DEV_BW_X(name) \
void CUDA16::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(half, x), CDATA(half, y), CDATA(half, gy), size, \
      MDATA(half, gx)); \
}

#define CUDA16_DEV_FW_X_CONST(name) \
void CUDA16::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(half, x), k, size, MDATA(half, y)); \
}

#define CUDA16_DEV_BW_X_CONST(name) \
void CUDA16::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(half, x), CDATA(half, y), CDATA(half, gy), k, size, \
      MDATA(half, gx)); \
}

#define CUDA16_DEV_FW_X_SCALAR(name) \
void CUDA16::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(half, x), CDATA(half, k), size, \
      x.shape().has_batch(), k.shape().has_batch(), MDATA(half, y)); \
}

#define CUDA16_DEV_FW_AB(name) \
void CUDA16::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(half, a), CDATA(half, b), size, \
      a.shape().has_batch(), b.shape().has_batch(), MDATA(half, y)); \
}

#define CUDA16_DEV_BW_AB(name) \
void CUDA16::name##_bw_impl( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(half, a), CDATA(half, b), CDATA(half, y), CDATA(half, gy), size, \
      a.shape().has_batch(), b.shape().has_batch(), \
      MDATA(half, ga), MDATA(half, gb)); \
}

#endif  // PRIMITIV_DEVICES_CUDA16_OPS_COMMON_H_
