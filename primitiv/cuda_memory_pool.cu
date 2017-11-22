#include <config.h>

#include <iostream>
#include <primitiv/cuda_memory_pool.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>
#include <primitiv/numeric_utils.h>

using std::cerr;
using std::endl;
using std::make_pair;

namespace primitiv {

#ifdef PRIMITIV_NEED_EXPLICIT_STATIC_SYMBOLS
template<>
  std::uint64_t mixins::Identifiable<CUDAMemoryPool>::next_id_ = 0;
template<>
  std::unordered_map<std::uint64_t, CUDAMemoryPool *>
  mixins::Identifiable<CUDAMemoryPool>::objects_;
template<>
  std::mutex mixins::Identifiable<CUDAMemoryPool>::mutex_;
#endif  // PRIMITIV_NEED_EXPLICIT_STATIC_SYMBOLS

CUDAMemoryPool::CUDAMemoryPool(std::uint32_t device_id)
: dev_id_(device_id)
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
}

CUDAMemoryPool::~CUDAMemoryPool() {
  // NOTE(odashi):
  // Due to GC-based languages, we chouldn't assume that all memories were
  // disposed before arriving this code.
  while (!supplied_.empty()) {
    free(supplied_.begin()->first);
  }
  release_reserved_blocks();
}

std::shared_ptr<void> CUDAMemoryPool::allocate(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t), "");

  static const std::uint64_t MAX_SHIFTS = 63;
  const std::uint64_t shift = numeric_utils::calculate_shifts(size);
  if (shift > MAX_SHIFTS) THROW_ERROR("Invalid memory size: " << size);

  void *ptr;
  if (reserved_[shift].empty()) {
    // Allocates a new block.
    CUDA_CALL(::cudaSetDevice(dev_id_));
    if (::cudaMalloc(&ptr, 1ull << shift) != ::cudaSuccess) {
      // Maybe out-of-memory.
      // Release other blocks and try allocation again.
      release_reserved_blocks();
      CUDA_CALL(::cudaMalloc(&ptr, 1ull << shift));
    }
    supplied_.emplace(ptr, shift);
  } else {
    // Returns an existing block.
    ptr = reserved_[shift].back();
    reserved_[shift].pop_back();
    supplied_.emplace(ptr, shift);
  }

  return std::shared_ptr<void>(ptr, CUDAMemoryDeleter(id()));
}

void CUDAMemoryPool::free(void *ptr) {
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
