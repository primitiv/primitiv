#include <primitiv/config.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda.h>

#include <primitiv/cuda_device.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>
#include <primitiv/shape_ops.h>
#include <primitiv/tensor.h>

#define CUDA_CALL(f) { \
  ::cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    PRIMITIV_THROW_ERROR( \
        "CUDA function failed.\n statement: " << #f \
        << "\n  error: " << err); \
  } \
}

#define CDATA(x) static_cast<const float *>(x.unsafe_handle())
#define MDATA(x) static_cast<float *>(x.unsafe_mutable_handle())

namespace {

primitiv::devices::CUDA *device = nullptr;
std::size_t device_id;
std::size_t max_x_size;

void check_device(const primitiv::Tensor &x) {
  primitiv::Device *dev = &x.device();
  if (dev != ::device) {
    PRIMITIV_THROW_ERROR("Device mismatched: " << dev << " != " << ::device);
  }
}

std::size_t calc_grid_size(std::size_t num_total_threads) {
  return (num_total_threads + max_x_size - 1) / max_x_size;
}

__global__ void forward_device(
    const float *x0_v, bool x0_b,
    const float *x1_v, bool x1_b,
    float *y0_v, bool y0_b,
    std::size_t volume) {
  static_cast<void>(x0_v); static_cast<void>(x0_b);
  static_cast<void>(x1_v); static_cast<void>(x1_b);
  static_cast<void>(y0_v); static_cast<void>(y0_b);
  const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  const std::size_t j = blockIdx.y * volume;
  if (i < volume) {
    y0_v[i + j * y0_b] = x0_v[i + j * x0_b] + x1_v[i + j * x1_b];
  }
}

__global__ void backward_device(
    const float *x0_v, float *x0_g, bool x0_b,
    const float *x1_v, float *x1_g, bool x1_b,
    const float *y0_v, const float *y0_g, bool y0_b,
    const std::size_t volume) {
  static_cast<void>(x0_v); static_cast<void>(x0_g); static_cast<void>(x0_b);
  static_cast<void>(x1_v); static_cast<void>(x1_g); static_cast<void>(x1_b);
  static_cast<void>(y0_v); static_cast<void>(y0_g); static_cast<void>(y0_b);
  const std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  const std::size_t j = blockIdx.y * volume;
  if (i < volume) {
    ::atomicAdd(x0_g + i + j * x0_b, y0_g[i + j * y0_b]);
    ::atomicAdd(x1_g + i + j * x1_b, y0_g[i + j * y0_b]);
  }
}

}  // namespace

extern "C" {

void set_device(primitiv::Device *dev) {
  const auto dev_type = dev->type();
  if (dev_type != primitiv::Device::DeviceType::CUDA) {
    PRIMITIV_THROW_ERROR(
        "Invalid device type: " << static_cast<std::uint32_t>(dev_type));
  }
  ::device = static_cast<primitiv::devices::CUDA *>(dev);
  ::device_id = ::device->device_id();
  ::max_x_size = ::device->max_x_size();
}

std::size_t get_argn() { return 2; }
std::size_t get_retn() { return 1; }

void forward_shape(const primitiv::Shape *args, primitiv::Shape *rets) {
  static_cast<void>(args);
  static_cast<void>(rets);
  rets[0] = primitiv::shape_ops::elementwise(args[0], args[1]);
}

void forward(const primitiv::Tensor *args, primitiv::Tensor *rets) {
  std::size_t batch = 0;
  for (std::size_t i = 0; i < ::get_argn(); ++i) {
    ::check_device(args[i]);
  }
  for (std::size_t i = 0; i < ::get_retn(); ++i) {
    ::check_device(rets[i]);
    batch = std::max<std::size_t>(batch, rets[i].shape().batch());
  }
  const std::size_t volume = rets[0].shape().volume();
  const std::size_t grid_size = ::calc_grid_size(volume);
  CUDA_CALL(::cudaSetDevice(::device_id));
  ::forward_device<<<grid_size, ::max_x_size>>>(
      CDATA(args[0]), args[0].shape().batch(),
      CDATA(args[1]), args[1].shape().batch(),
      MDATA(rets[0]), rets[0].shape().batch(),
      volume);
}

void backward(
    const primitiv::Tensor *args_v, const primitiv::Tensor *rets_v,
    const primitiv::Tensor *rets_g, primitiv::Tensor *args_g) {
  std::size_t batch = 0;
  for (std::size_t i = 0; i < ::get_argn(); ++i) {
    ::check_device(args_v[i]);
    ::check_device(args_g[i]);
  }
  for (std::size_t i = 0; i < ::get_retn(); ++i) {
    ::check_device(rets_v[i]);
    ::check_device(rets_g[i]);
    batch = std::max<std::size_t>(batch, rets_v[i].shape().batch());
  }
  const std::size_t volume= rets_v[0].shape().volume();
  const std::size_t grid_size = ::calc_grid_size(volume);
  CUDA_CALL(::cudaSetDevice(::device_id));
  ::backward_device<<<grid_size, ::max_x_size>>>(
      CDATA(args_v[0]), MDATA(args_g[0]), args_v[0].shape().batch(),
      CDATA(args_v[1]), MDATA(args_g[1]), args_v[1].shape().batch(),
      CDATA(rets_v[0]), CDATA(rets_g[0]), rets_v[0].shape().batch(),
      volume);
}

}  // extern "C"
