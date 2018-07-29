#ifndef PRIMITIV_DEVICES_CUDA_OPS_COMMON_H_
#define PRIMITIV_DEVICES_CUDA_OPS_COMMON_H_

/*
 * Macros for device functions.
 */

#define IDX (threadIdx.x + blockIdx.x * blockDim.x)
#define IDY (threadIdx.y + blockIdx.y * blockDim.y)

#define FLOAT_POSITIVE_INFINITY ::__uint_as_float(0x7f800000)
#define FLOAT_NEGATIVE_INFINITY ::__uint_as_float(0xff800000)

#define CUDADEV_KERNEL_FW_X(name, op) \
__global__ void name##_fw_dev(const float *px, std::uint32_t size, float *py) { \
  const std::uint32_t i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDADEV_KERNEL_BW_X(name, op) \
__global__ void name##_bw_dev( \
    const float *px, const float *py, const float *pgy, std::uint32_t size, \
    float *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const std::uint32_t i = IDX; \
  if (i < size) pgx[i] += (op); \
}

#define CUDADEV_KERNEL_FW_X_CONST(name, op) \
__global__ void name##_fw_dev( \
    const float *px, float k, std::uint32_t size, float *py) { \
  const std::uint32_t i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDADEV_KERNEL_BW_X_CONST(name, op) \
__global__ void name##_bw_dev( \
    const float *px, const float *py, const float *pgy, float k, \
    std::uint32_t size, float *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const std::uint32_t i = IDX; \
  if (i < size) pgx[i] += (op); \
}

#define CUDADEV_KERNEL_FW_X_SCALAR_R(name, op) \
__global__ void name##_fw_dev( \
    const float *px, const float *pk, std::uint32_t size, std::uint32_t mbx, \
    std::uint32_t mbk, float *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(px[i + mbx * shift], pk[mbk * blockIdx.y]); \
}

#define CUDADEV_KERNEL_FW_X_SCALAR_L(name, op) \
__global__ void name##_fw_dev( \
    const float *px, const float *pk, std::uint32_t size, std::uint32_t mbx, \
    std::uint32_t mbk, float *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pk[mbk * blockIdx.y], px[i + mbx * shift]); \
}

#define CUDADEV_KERNEL_FW_AB(name, op) \
__global__ void name##_fw_dev( \
    const float *pa, const float *pb, std::uint32_t size, std::uint32_t mba, \
    std::uint32_t mbb, float *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pa[i + mba * shift], pb[i + mbb * shift]); \
}

/*
 * Macros for primitiv::Device overrides.
 */

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define CDATA(x) static_cast<const float *>(get_handle(x))
#define MDATA(x) static_cast<float *>(get_mutable_handle(x))

#define CUDADEV_FW_X(name) \
void CUDA::name##_fw_impl(const Tensor &x, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), size, MDATA(y)); \
}

#define CUDADEV_BW_X(name) \
void CUDA::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(x), CDATA(y), CDATA(gy), size, MDATA(gx)); \
}

#define CUDADEV_FW_X_CONST(name) \
void CUDA::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size,dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), k, size, MDATA(y)); \
}

#define CUDADEV_BW_X_CONST(name) \
void CUDA::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(x), CDATA(y), CDATA(gy), k, size, MDATA(gx)); \
}

#define CUDADEV_FW_X_SCALAR(name) \
void CUDA::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(x), CDATA(k), size, \
      x.shape().has_batch(), k.shape().has_batch(), MDATA(y)); \
}

#define CUDADEV_FW_AB(name) \
void CUDA::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(a), CDATA(b), size, \
      a.shape().has_batch(), b.shape().has_batch(), MDATA(y)); \
}

#define CUDADEV_BW_AB(name) \
void CUDA::name##_bw_impl( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(a), CDATA(b), CDATA(y), CDATA(gy), size, \
      a.shape().has_batch(), b.shape().has_batch(), MDATA(ga), MDATA(gb)); \
}

#endif  // PRIMITIV_DEVICES_CUDA_OPS_COMMON_H_
