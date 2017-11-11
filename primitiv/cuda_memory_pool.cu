#include <config.h>

#include <iostream>
#include <primitiv/cuda_memory_pool.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>

using std::cerr;
using std::endl;
using std::make_pair;

namespace primitiv {

std::size_t CUDAMemoryPool::next_pool_id_ = 0;
std::unordered_map<std::size_t, CUDAMemoryPool *> CUDAMemoryPool::pools_;

CUDAMemoryPool::CUDAMemoryPool(std::uint32_t device_id)
: pool_id_(next_pool_id_++)
, dev_id_(device_id)
, reserved_(64)
, supplied_() {
  // Retrieves device properties.
  int max_devs;
  CUDA_CALL(::cudaGetDeviceCount(&max_devs));
  if (dev_id_ >= static_cast<std::uint32_t>(max_devs)) {
    THROW_ERROR(
        "Invalid CUDA device ID. given: " << dev_id_
        << " >= #devices: " << max_devs);
  }

  // Registers this object.
  pools_.insert(std::make_pair(pool_id_, this));
}

CUDAMemoryPool::~CUDAMemoryPool() {
  // Unregisters this object.
  pools_.erase(pools_.find(pool_id_));

  // NOTE(odashi):
  // Due to GC-based languages, we chouldn't assume that all memories were
  // disposed before arriving this code.
  while (!supplied_.empty()) {
    free_inner(supplied_.begin()->first);
  }
  release_reserved_blocks();
}

std::shared_ptr<void> CUDAMemoryPool::allocate(std::size_t size) {
  static const std::uint32_t MAX_SCALE = 63;
  std::uint32_t scale = 0;
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
    if (::cudaMalloc(&ptr, 1ull << scale) != ::cudaSuccess) {
      // Maybe out-of-memory.
      // Release other blocks and try allocation again.
      release_reserved_blocks();
      CUDA_CALL(::cudaMalloc(&ptr, 1ull << scale));
    }
    supplied_.insert(make_pair(ptr, scale));
  } else {
    // Returns an existing block.
    ptr = reserved_[scale].back();
    reserved_[scale].pop_back();
    supplied_.insert(make_pair(ptr, scale));
  }

  return std::shared_ptr<void>(ptr, CUDAMemoryDeleter(pool_id_));
}

void CUDAMemoryPool::free(std::size_t pool_id, void *ptr) {
  auto it = pools_.find(pool_id);
  if (it != pools_.end()) {
    // Found a corresponding pool object, delete ptr.
    it->second->free_inner(ptr);
  }
  // Otherwise, ptr is assumed as to be deleted before calling this function.
}

void CUDAMemoryPool::free_inner(void *ptr) {
  auto it = supplied_.find(ptr);
  if (it == supplied_.end()) {
    THROW_ERROR("Detected to dispose unknown handle: " << ptr);
  }

  reserved_[it->second].emplace_back(ptr);
  supplied_.erase(it);
}

void CUDAMemoryPool::release_reserved_blocks() {
  for (auto &ptrs : reserved_) {
    while (!ptrs.empty()) {
      CUDA_CALL(::cudaFree(ptrs.back()));
      ptrs.pop_back();
    }
  }
}

}  // namespace
