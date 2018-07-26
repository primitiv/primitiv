#include <primitiv/config.h>

#include <iostream>

#include <primitiv/core/error.h>
#include <primitiv/core/memory_pool.h>
#include <primitiv/core/numeric_utils.h>

using std::cerr;
using std::endl;
using std::make_pair;

namespace primitiv {

MemoryPool::MemoryPool(
    std::function<void *(std::size_t)> allocator,
    std::function<void(void *)> deleter)
: allocator_(allocator)
, deleter_(deleter)
, reserved_(64)
, supplied_() {}

MemoryPool::~MemoryPool() {
  // NOTE(odashi):
  // Due to GC-based languages, we chouldn't assume that all memories were
  // disposed before arriving this code.
  while (!supplied_.empty()) {
    free(supplied_.begin()->first);
  }
  release_reserved_blocks();
}

std::shared_ptr<void> MemoryPool::allocate(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t), "");

  if (size == 0) return std::shared_ptr<void>();

  static const std::uint64_t MAX_SHIFTS = 63;
  const std::uint64_t shift = numeric_utils::calculate_shifts(size);
  if (shift > MAX_SHIFTS) PRIMITIV_THROW_ERROR("Invalid memory size: " << size);

  void *ptr;
  if (reserved_[shift].empty()) {
    // Allocates a new block.
    try {
      ptr = allocator_(1ull << shift);
    } catch (...) {
      // Maybe out-of-memory.
      // Release other blocks and try allocation again.
      release_reserved_blocks();
      // Below allocation may throw an error when the memory allocation
      // process finally failed.
      ptr = allocator_(1ull << shift);
    }
    supplied_.emplace(ptr, shift);
  } else {
    // Returns an existing block.
    ptr = reserved_[shift].back();
    reserved_[shift].pop_back();
    supplied_.emplace(ptr, shift);
  }

  return std::shared_ptr<void>(ptr, Deleter(id()));
}

void MemoryPool::free(void *ptr) {
  auto it = supplied_.find(ptr);
  if (it == supplied_.end()) {
    PRIMITIV_THROW_ERROR("Detected to dispose unknown handle: " << ptr);
  }
  reserved_[it->second].emplace_back(ptr);
  supplied_.erase(it);
}

void MemoryPool::release_reserved_blocks() {
  for (auto &ptrs : reserved_) {
    while (!ptrs.empty()) {
      deleter_(ptrs.back());
      ptrs.pop_back();
    }
  }
}

}  // namespace
