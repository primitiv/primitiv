#ifndef PRIMITIV_INTERNAL_CUDA_UTILS_H_
#define PRIMITIV_INTERNAL_CUDA_UTILS_H_

#include <primitiv/config.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#ifdef PRIMITIV_USE_CUDNN
#include <cudnn.h>
#endif  // PRIMITIV_USE_CUDNN
#include <curand.h>

#include <primitiv/core/memory_pool.h>
#include <primitiv/core/mixins/nonmovable.h>

#define CUDA_CALL(f) { \
  ::cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    PRIMITIV_THROW_ERROR( \
        "CUDA function failed.\n  statement: " << #f \
        << "\n  error: " << err \
        << ": " << ::cudaGetErrorString(err)); \
  } \
}

#define CUBLAS_CALL(f) { \
  ::cublasStatus_t err = (f); \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    PRIMITIV_THROW_ERROR( \
        "CUBLAS function failed.\n  statement: " << #f \
        << "\n  error: " << err \
        << ": " << primitiv::cuda::cublasGetErrorString(err)); \
  } \
}

#define CURAND_CALL(f) { \
  ::curandStatus_t err = (f); \
  if (err != CURAND_STATUS_SUCCESS) { \
    PRIMITIV_THROW_ERROR( \
        "CURAND function failed.\n  statement: " << #f \
        << "\n  error: " << err \
        << ": " << primitiv::cuda::curandGetErrorString(err)); \
  } \
}

#ifdef PRIMITIV_USE_CUDNN

#define CUDNN_CALL(f) { \
  ::cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    PRIMITIV_THROW_ERROR( \
        "CUDNN function failed.\n  statement: " << #f \
        << "\n  error: " << err \
        << ": " << ::cudnnGetErrorString(err)); \
  } \
}

#endif  // PRIMITIV_USE_CUDNN

namespace primitiv {
namespace cuda {

/**
 * Retrieves cuBLAS error string.
 * @param err cuBLAS error cude.
 * @return Error string.
 */
std::string cublasGetErrorString(::cublasStatus_t err);

/**
 * Retrieves cuRAND error string.
 * @param err cuRAND error cude.
 * @return Error string.
 */
std::string curandGetErrorString(::curandStatus_t err);

/**
 * CuBLAS initializer/finalizer.
 */
class CuBLASHandle : primitiv::mixins::Nonmovable<CuBLASHandle> {
public:
  explicit CuBLASHandle(std::uint32_t dev_id) {
    CUDA_CALL(::cudaSetDevice(dev_id));
    CUBLAS_CALL(::cublasCreate(&handle_));
  }

  ~CuBLASHandle() {
    try {
      CUBLAS_CALL(::cublasDestroy(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuBLASHandle::~CuBLASHandle()`"
        << std::endl;
      std::abort();
    }
  }

  ::cublasHandle_t get() const { return handle_; }

private:
  ::cublasHandle_t handle_ = NULL;
};

/**
 * CuRAND initializer/finalizer.
 */
class CuRANDHandle : primitiv::mixins::Nonmovable<CuRANDHandle> {
public:
  CuRANDHandle(std::uint32_t dev_id, std::uint32_t rng_seed) {
    CUDA_CALL(::cudaSetDevice(dev_id));
    CURAND_CALL(::curandCreateGenerator(&handle_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(::curandSetPseudoRandomGeneratorSeed(handle_, rng_seed));
  }

  ~CuRANDHandle() {
    try {
      CURAND_CALL(::curandDestroyGenerator(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuRANDHandle::~CuRANDHandle()`"
        << std::endl;
      std::abort();
    }
  }

  ::curandGenerator_t get() const { return handle_; }

private:
  ::curandGenerator_t handle_ = NULL;
};

#ifdef PRIMITIV_USE_CUDNN

/**
 * cuDNN initializer/finalizer.
 */
class CuDNNHandle : primitiv::mixins::Nonmovable<CuDNNHandle> {
public:
  explicit CuDNNHandle(std::uint32_t dev_id) {
    CUDA_CALL(::cudaSetDevice(dev_id));
    CUDNN_CALL(::cudnnCreate(&handle_));
  }

  ~CuDNNHandle() {
    try {
      CUDNN_CALL(::cudnnDestroy(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuDNNHandle::~CuDNNHandle()`"
        << std::endl;
      std::abort();
    }
  }

  ::cudnnHandle_t get() const { return handle_; }

private:
  ::cudnnHandle_t handle_ = NULL;
};

/**
 * Wrapper of cudnnTensorDescriptor_t.
 */
class CuDNNTensorDescriptor
: public primitiv::mixins::Nonmovable<CuDNNTensorDescriptor> {
public:
  CuDNNTensorDescriptor(
      std::uint32_t n, std::uint32_t c, std::uint32_t h, std::uint32_t w,
      ::cudnnDataType_t data_type) {
    CUDNN_CALL(::cudnnCreateTensorDescriptor(&handle_));
    CUDNN_CALL(::cudnnSetTensor4dDescriptor(
          handle_, CUDNN_TENSOR_NCHW, data_type, n, c, h, w));
  }

  ~CuDNNTensorDescriptor() {
    try {
      CUDNN_CALL(::cudnnDestroyTensorDescriptor(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuDNNTensorDescriptor::~CuDNNTensorDescriptor()`"
        << std::endl;
      std::abort();
    }
  }

  ::cudnnTensorDescriptor_t get() const { return handle_; }

private:
  ::cudnnTensorDescriptor_t handle_ = NULL;
};

/**
 * Wrapper of cudnnFilterDescriptor_t.
 */
class CuDNNFilterDescriptor
: public primitiv::mixins::Nonmovable<CuDNNFilterDescriptor> {
public:
  CuDNNFilterDescriptor(
      std::uint32_t k, std::uint32_t c, std::uint32_t h, std::uint32_t w,
      ::cudnnDataType_t data_type) {
    CUDNN_CALL(::cudnnCreateFilterDescriptor(&handle_));
    CUDNN_CALL(::cudnnSetFilter4dDescriptor(
          handle_, data_type, CUDNN_TENSOR_NCHW, k, c, h, w));
  }

  ~CuDNNFilterDescriptor() {
    try {
      CUDNN_CALL(::cudnnDestroyFilterDescriptor(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuDNNFilterDescriptor::~CuDNNFilterDescriptor()`"
        << std::endl;
      std::abort();
    }
  }

  ::cudnnFilterDescriptor_t get() const { return handle_; }

private:
  ::cudnnFilterDescriptor_t handle_ = NULL;
};

/**
 * Wrapper of cudnnConvolutionDescriptor_t.
 */
class CuDNNConvolutionDescriptor
: public primitiv::mixins::Nonmovable<CuDNNConvolutionDescriptor> {
public:
  CuDNNConvolutionDescriptor(
      std::uint32_t padding_h, std::uint32_t padding_w,
      std::uint32_t stride_h, std::uint32_t stride_w,
      std::uint32_t dilation_h, std::uint32_t dilation_w,
      ::cudnnDataType_t data_type) {
    CUDNN_CALL(::cudnnCreateConvolutionDescriptor(&handle_));
#if CUDNN_MAJOR >= 6
    CUDNN_CALL(::cudnnSetConvolution2dDescriptor(
          handle_,
          padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
          CUDNN_CONVOLUTION, data_type));
#else
    static_cast<void>(data_type);  // unused
    if (dilation_h > 1 || dilation_w > 1) {
      PRIMITIV_THROW_NOT_IMPLEMENTED_WITH_MESSAGE(
          "Dilated convolution is supported by cuDNN 6.0 or later.");
    }
    CUDNN_CALL(::cudnnSetConvolution2dDescriptor(
          handle_,
          padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
          CUDNN_CONVOLUTION));
#endif  // CUDNN_MAJOR
  }

  ~CuDNNConvolutionDescriptor() {
    try {
      CUDNN_CALL(::cudnnDestroyConvolutionDescriptor(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuDNNConvolutionDescriptor::~CuDNNConvolutionDescriptor()`"
        << std::endl;
      std::abort();
    }
  }

  ::cudnnConvolutionDescriptor_t get() const { return handle_; }

private:
  ::cudnnConvolutionDescriptor_t handle_ = NULL;
};

/**
 * Wrapper of cudnnPoolingDescriptor_t.
 */
class CuDNNPoolingDescriptor
: public primitiv::mixins::Nonmovable<CuDNNPoolingDescriptor> {
public:
  CuDNNPoolingDescriptor(
      ::cudnnPoolingMode_t mode,
      std::uint32_t window_h, std::uint32_t window_w,
      std::uint32_t padding_h, std::uint32_t padding_w,
      std::uint32_t stride_h, std::uint32_t stride_w) {
    CUDNN_CALL(::cudnnCreatePoolingDescriptor(&handle_));
    CUDNN_CALL(::cudnnSetPooling2dDescriptor(
          handle_, mode, CUDNN_PROPAGATE_NAN,
          window_h, window_w, padding_h, padding_w, stride_h, stride_w));
  }

  ~CuDNNPoolingDescriptor() {
    try {
      CUDNN_CALL(::cudnnDestroyPoolingDescriptor(handle_));
    } catch (...) {
      std::cerr
        << "Unexpected exception occurred at "
        << "`CuDNNPoolingDescriptor::~CuDNNPoolingDescriptor()`"
        << std::endl;
      std::abort();
    }
  }

  ::cudnnPoolingDescriptor_t get() const { return handle_; }

private:
  ::cudnnPoolingDescriptor_t handle_ = NULL;
};

#endif  // PRIMITIV_USE_CUDNN

/**
 * Hidden objects of CUDA devices.
 */
struct InternalState {
  InternalState(std::uint32_t dev_id, std::uint32_t rng_seed)
    : cublas(dev_id)
    , curand(dev_id, rng_seed)
#ifdef PRIMITIV_USE_CUDNN
    , cudnn(dev_id)
#endif  // PRIMITIV_USE_CUDNN
    , pool(
        [dev_id](std::size_t size) -> void * {  // allocator
          void *ptr;
          CUDA_CALL(::cudaSetDevice(dev_id));
          CUDA_CALL(::cudaMalloc(&ptr, size));
          return ptr;
        },
        [](void *ptr) -> void {  // deleter
          CUDA_CALL(::cudaFree(ptr));
        }) {}
  CuBLASHandle cublas;
  CuRANDHandle curand;
#ifdef PRIMITIV_USE_CUDNN
  CuDNNHandle cudnn;
#endif  // PRIMITIV_USE_CUDNN
  MemoryPool pool;
  ::cudaDeviceProp prop;
};

}  // namespace cuda
}  // namespace primitiv

#endif  // PRIMITIV_INTERNAL_CUDA_UTILS_H_
