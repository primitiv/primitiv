#ifndef PRIMITIV_CUDA_MEMORY_POOL_H_
#define PRIMITIV_CUDA_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <primitiv/mixins.h>

namespace primitiv {

class CUDAMemoryDeleter;

/**
 * Memory manager on the CUDA devices.
 */
class CUDAMemoryPool : public mixins::Identifiable<CUDAMemoryPool> {
  friend CUDAMemoryDeleter;

  CUDAMemoryPool() = delete;

public:
  /**
   * Creates a memory pool.
   * @param device_id CUDA Device ID on which memories are stored.
   */
  explicit CUDAMemoryPool(std::uint32_t device_id);

  ~CUDAMemoryPool();

  /**
   * Allocates a memory.
   * @param size Size of the resulting memory.
   * @return Shared pointer of the allocated memory.
   */
  std::shared_ptr<void> allocate(std::size_t size);

private:
  /**
   * Disposes the memory managed by this pool.
   * @param ptr Handle of the memory to be disposed.
   */
  void free(void *ptr);

  /**
   * Releases all reserved memory blocks.
   */
  void release_reserved_blocks();

  std::uint32_t dev_id_;
  std::vector<std::vector<void *>> reserved_;
  std::unordered_map<void *, std::uint32_t> supplied_;
};

/**
 * Custom deleter class for CUDA memories.
 */
class CUDAMemoryDeleter {
  CUDAMemoryDeleter() = delete;
public:
  explicit CUDAMemoryDeleter(std::uint64_t pool_id) : pool_id_(pool_id) {}

  void operator()(void *ptr) {
    try {
      CUDAMemoryPool::get_object(pool_id_).free(ptr);
    } catch (primitiv::Error) {
      // Memory pool already has gone and the pointer is already deleted by
      // the memory pool.
    }
  }

private:
  std::uint64_t pool_id_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CUDA_MEMORY_POOL_H_
