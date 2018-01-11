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

/*
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
  ::cublasHandle_t handle_;
};

/*
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
  ::curandGenerator_t handle_;
};

/*
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
  ::cudnnHandle_t handle_;
};

}  // namespace cuda
}  // namespace primitiv

#endif  // PRIMITIV_CUDA_UTILS_H_
