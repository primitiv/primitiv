#include <config.h>

#include <iostream>
#include <primitiv/cuda_memory_pool.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>

using std::cerr;
using std::endl;
using std::make_pair;
using std::vector;

namespace primitiv {

CUDAMemoryPool::CUDAMemoryPool(unsigned device_id)
: dev_id_(device_id)
, reserved_(64)
, supplied_() {
  // Retrieves device properties.
  int max_devs;
  CUDA_CALL(::cudaGetDeviceCount(&max_devs));
  if (dev_id_ >= static_cast<unsigned>(max_devs)) {
    THROW_ERROR(
        "Invalid CUDA device ID. given: " << dev_id_ << " >= " << max_devs);
  }
}

CUDAMemoryPool::~CUDAMemoryPool() {
  if (!supplied_.empty()) {
    cerr << "FATAL ERROR: Detected memory leak on CUDA device!" << endl;
    cerr << "Leaked blocks (handle: size):" << endl;
    for (const auto &kv : supplied_) {
      cerr << "  " << kv.first << ": " << kv.second << endl;
    }
    std::abort();
  }

  for (auto &ptrs : reserved_) {
    for (void *ptr : ptrs) {
      CUDA_CALL(::cudaFree(ptr));
    }
  }
}

void *CUDAMemoryPool::allocate(unsigned size) {
  unsigned i = 0;
  while (1u << i < size) ++i;

  if (reserved_[i].empty()) {
    // Allocates a new block.
    void *ptr;
    CUDA_CALL(::cudaSetDevice(dev_id_));
    CUDA_CALL(::cudaMalloc(&ptr, 1u << i));
    supplied_.insert(make_pair(ptr, i));
    return ptr;
  }

  // Returns an existing block.
  void *ptr = reserved_[i].back();
  reserved_[i].pop_back();
  supplied_.insert(make_pair(ptr, i));
  return ptr;
}

void CUDAMemoryPool::free(void *ptr) {
  auto it = supplied_.find(ptr);
  if (it == supplied_.end()) {
    THROW_ERROR("Detected to dispose unknown handle: " << ptr);
  }

  reserved_[it->second].emplace_back(ptr);
  supplied_.erase(it);
}

}  // namespace
