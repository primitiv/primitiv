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
        "Invalid CUDA device ID. given: " << dev_id_
        << " >= #devices: " << max_devs);
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

std::shared_ptr<void> CUDAMemoryPool::allocate(std::uint64_t size) {
  static const unsigned MAX_SCALE = 63;
  unsigned scale = 0;
  while (1ull << scale < size) {
    if (scale == MAX_SCALE) {
      THROW_ERROR(
          "Attempted to allocate more than 2^" << MAX_SCALE << " bytes.");
    }
    ++scale;
  }

  void *ptr;
  if (reserved_[scale].empty()) {
    // Allocates a new block.
    CUDA_CALL(::cudaSetDevice(dev_id_));
    CUDA_CALL(::cudaMalloc(&ptr, 1ull << scale));
    supplied_.insert(make_pair(ptr, scale));
  } else {
    // Returns an existing block.
    ptr = reserved_[scale].back();
    reserved_[scale].pop_back();
    supplied_.insert(make_pair(ptr, scale));
  }

  return std::shared_ptr<void>(ptr, CUDAMemoryDeleter(*this));
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
