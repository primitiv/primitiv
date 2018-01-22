#ifndef PRIMITIV_CUDA_UTILS_H_
#define PRIMITIV_CUDA_UTILS_H_

#include <string>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <curand.h>

#include <primitiv/mixins.h>

#define CUDA_CALL(f) { \
  ::cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    THROW_ERROR( \
        "CUDA function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << ::cudaGetErrorString(err)); \
  } \
}

#define CUBLAS_CALL(f) { \
  ::cublasStatus_t err = (f); \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    THROW_ERROR( \
        "CUBLAS function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << primitiv::cuda::cublasGetErrorString(err)); \
  } \
}

#define CURAND_CALL(f) { \
  ::curandStatus_t err = (f); \
  if (err != CURAND_STATUS_SUCCESS) { \
    THROW_ERROR( \
        "CURAND function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << primitiv::cuda::curandGetErrorString(err)); \
  } \
}

#define CUDNN_CALL(f) { \
  ::cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    THROW_ERROR( \
        "CUDNN function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << ::cudnnGetErrorString(err)); \
  } \
}

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
    CUBLAS_CALL(::cublasDestroy(handle_));
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
    CURAND_CALL(::curandDestroyGenerator(handle_));
  }

  ::curandGenerator_t get() const { return handle_; }

private:
  ::curandGenerator_t handle_ = NULL;
};

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
    CUDNN_CALL(::cudnnDestroy(handle_));
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
      std::uint32_t n, std::uint32_t c, std::uint32_t h, std::uint32_t w) {
    CUDNN_CALL(::cudnnCreateTensorDescriptor(&handle_));
    CUDNN_CALL(::cudnnSetTensor4dDescriptor(
          handle_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
  }

  ~CuDNNTensorDescriptor() {
    CUDNN_CALL(::cudnnDestroyTensorDescriptor(handle_));
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
      std::uint32_t k, std::uint32_t c, std::uint32_t h, std::uint32_t w) {
    CUDNN_CALL(::cudnnCreateFilterDescriptor(&handle_));
    CUDNN_CALL(::cudnnSetFilter4dDescriptor(
          handle_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, h, w));
  }

  ~CuDNNFilterDescriptor() {
    CUDNN_CALL(::cudnnDestroyFilterDescriptor(handle_));
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
      std::uint32_t dilation_h, std::uint32_t dilation_w) {
    CUDNN_CALL(::cudnnCreateConvolutionDescriptor(&handle_));
#if CUDNN_MAJOR >= 6
    CUDNN_CALL(::cudnnSetConvolution2dDescriptor(
          handle_,
          padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
          CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
#else
    if (dilation_h > 1 || dilation_w > 1) {
      THROW_NOT_IMPLEMENTED_WITH_MESSAGE(
          "Dilated convolution is supported by cuDNN 6.0 or later.");
    }
    CUDNN_CALL(::cudnnSetConvolution2dDescriptor(
          handle_,
          padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
          CUDNN_CONVOLUTION));
#endif  // CUDNN_MAJOR
  }

  ~CuDNNConvolutionDescriptor() {
    CUDNN_CALL(::cudnnDestroyConvolutionDescriptor(handle_));
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
    CUDNN_CALL(::cudnnDestroyPoolingDescriptor(handle_));
  }

  ::cudnnPoolingDescriptor_t get() const { return handle_; }

private:
  ::cudnnPoolingDescriptor_t handle_ = NULL;
};

}  // namespace cuda
}  // namespace primitiv

#endif  // PRIMITIV_CUDA_UTILS_H_
