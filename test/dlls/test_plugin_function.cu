// NOTE(odashi):
// This code contains a sample implementation of a precompiled function on the
// CUDA device that performs the elementwise addition: `y0 = x0 + x1`.

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

#define CDATA(x) static_cast<const float *>((x)->unsafe_handle())
#define MDATA(x) static_cast<float *>((x)->unsafe_mutable_handle())

namespace {

primitiv::devices::CUDA *device = nullptr;
std::size_t device_id;
std::size_t max_x_size;

void set_device(primitiv::Device &dev) {
  const auto dev_type = dev.type();
  if (dev_type != primitiv::Device::DeviceType::CUDA) {
    PRIMITIV_THROW_ERROR(
        "Invalid device type: " << static_cast<std::uint32_t>(dev_type));
  }
  ::device = static_cast<primitiv::devices::CUDA *>(&dev);
  ::device_id = ::device->device_id();
  ::max_x_size = ::device->max_x_size();
}

void check_device(const primitiv::Device &dev) {
  if (&dev != ::device) {
    PRIMITIV_THROW_ERROR("Device mismatched: " << &dev << " != " << ::device);
  }
}

std::size_t calc_grid_size(std::size_t num_total_threads) {
  return (num_total_threads + max_x_size - 1) / max_x_size;
}

__global__ void forward_device(
    const float *x0_v, std::uint32_t x0_b,
    const float *x1_v, std::uint32_t x1_b,
    float *y0_v, std::uint32_t y0_b,
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
    const float *x0_v, float *x0_g, std::uint32_t x0_b,
    const float *x1_v, float *x1_g, std::uint32_t x1_b,
    const float *y0_v, const float *y0_g, std::uint32_t y0_b,
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

std::size_t num_arguments() { return 2; }
std::size_t num_returns() { return 1; }

void forward_shape(
    const primitiv::Shape * const * const args,
    primitiv::Shape * const * const rets) {
  static_cast<void>(args);
  static_cast<void>(rets);
  *rets[0] = primitiv::shape_ops::elementwise(*args[0], *args[1]);
}

void forward(
    const primitiv::Tensor * const * const args,
    primitiv::Tensor * const * const rets) {
  static_cast<void>(args);
  static_cast<void>(rets);
  ::set_device(args[0]->device());
  ::check_device(args[1]->device());
  const primitiv::Shape arg0_s = args[0]->shape();
  const primitiv::Shape arg1_s = args[1]->shape();
  primitiv::Shape ret0_s;
  const primitiv::Shape * const args_sp[] {
    &arg0_s,
    &arg1_s,
  };
  primitiv::Shape * const rets_sp[] {
    &ret0_s,
  };
  forward_shape(args_sp, rets_sp);
  *rets[0] = ::device->new_raw_tensor(ret0_s);
  std::size_t batch = 0;
  batch = std::max<std::size_t>(batch, arg0_s.batch());
  batch = std::max<std::size_t>(batch, arg1_s.batch());
  const std::size_t volume = ret0_s.volume();
  const std::size_t grid_size = ::calc_grid_size(volume);
  CUDA_CALL(::cudaSetDevice(::device_id));
  ::forward_device<<<dim3(grid_size, batch), ::max_x_size>>>(
      CDATA(args[0]), arg0_s.has_batch(),
      CDATA(args[1]), arg1_s.has_batch(),
      MDATA(rets[0]), ret0_s.has_batch(),
      volume);
}

void backward(
    const primitiv::Tensor * const * const args_v,
    const primitiv::Tensor * const * const rets_v,
    const primitiv::Tensor * const * const rets_g,
    primitiv::Tensor * const * const args_g) {
  static_cast<void>(args_v);
  static_cast<void>(rets_v);
  static_cast<void>(rets_g);
  static_cast<void>(args_g);
  ::set_device(args_v[0]->device());
  ::check_device(args_v[1]->device());
  ::check_device(args_g[0]->device());
  ::check_device(args_g[1]->device());
  ::check_device(rets_v[0]->device());
  ::check_device(rets_g[0]->device());
  primitiv::Shape arg0_s = args_v[0]->shape();
  primitiv::Shape arg1_s = args_v[1]->shape();
  primitiv::Shape ret0_s = rets_v[0]->shape();
  std::size_t batch = 0;
  batch = std::max<std::size_t>(batch, arg0_s.batch());
  batch = std::max<std::size_t>(batch, arg1_s.batch());
  const std::size_t volume = ret0_s.volume();
  const std::size_t grid_size = ::calc_grid_size(volume);
  CUDA_CALL(::cudaSetDevice(::device_id));
  ::backward_device<<<dim3(grid_size, batch), ::max_x_size>>>(
      CDATA(args_v[0]), MDATA(args_g[0]), arg0_s.has_batch(),
      CDATA(args_v[1]), MDATA(args_g[1]), arg1_s.has_batch(),
      CDATA(rets_v[0]), CDATA(rets_g[0]), ret0_s.has_batch(),
      volume);
}

}  // extern "C"
