#include <config.h>

#include <primitiv/cuda_memory_pool.h>
#include <primitiv/error.h>

namespace primitiv {

CUDAMemoryPool::CUDAMemoryPool(unsigned device_id) : dev_id_(device_id) {
  THROW_ERROR("not implemeneted");
}

CUDAMemoryPool::~CUDAMemoryPool() {
  THROW_ERROR("not implemeneted");
}

void *CUDAMemoryPool::allocate(unsigned size) {
  THROW_ERROR("not implemeneted");
}

void CUDAMemoryPool::free(void *ptr) {
  THROW_ERROR("not implemeneted");
}

}  // namespace
