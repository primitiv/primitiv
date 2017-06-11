#ifndef PRIMITIV_CUDA_UTILS_H_
#define PRIMITIV_CUDA_UTILS_H_

#include <string>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>

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

}  // namespace cuda
}  // namespace primitiv

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


#endif  // PRIMITIV_CUDA_UTILS_H_
