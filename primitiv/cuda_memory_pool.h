#ifndef PRIMITIV_CUDA_MEMORY_POOL_H_
#define PRIMITIV_CUDA_MEMORY_POOL_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace primitiv {

class CUDAMemoryDeleter;

/**
 * Memory manager on the CUDA devices.
 */
class CUDAMemoryPool {
  friend CUDAMemoryDeleter;

  CUDAMemoryPool() = delete;
  CUDAMemoryPool(const CUDAMemoryPool &) = delete;
  CUDAMemoryPool(CUDAMemoryPool &&) = delete;
  CUDAMemoryPool &operator=(const CUDAMemoryPool &) = delete;
  CUDAMemoryPool &operator=(CUDAMemoryPool &&) = delete;

public:
  /**
   * Creates a memory pool.
   * @param device_id CUDA Device ID on which memories are stored.
   */
  explicit CUDAMemoryPool(unsigned device_id);

  ~CUDAMemoryPool();

  /**
   * Allocates a memory.
   * @param size Size of the resulting memory.
   * @return Shared pointer of the allocated memory.
   */
  std::shared_ptr<void> allocate(std::uint64_t size);

  /**
   * Retrieves pool ID.
   * @return pool ID.
   */
  std::uint64_t get_pool_id() const { return pool_id_; }

private:
  /**
   * Disposes the memory.
   * @param pool_id ID of the CUDAMemoryPool.
   * @param ptr Handle of the memory to be disposed.
   */
  static void free(std::uint64_t pool_id, void *ptr);

  /**
   * Disposes the memory managed by this pool.
   * @param ptr Handle of the memory to be disposed.
   */
  void free_inner(void *ptr);

  /**
   * Releases all reserved memory blocks.
   */
  void release_reserved_blocks();

  static std::uint64_t next_pool_id_;
  static std::unordered_map<std::uint64_t, CUDAMemoryPool *> pools_;

  std::uint64_t pool_id_;
  unsigned dev_id_;
  std::vector<std::vector<void *>> reserved_;
  std::unordered_map<void *, unsigned> supplied_;
};

/**
 * Custom deleter class for CUDA memories.
 */
class CUDAMemoryDeleter {
  CUDAMemoryDeleter() = delete;
public:
  explicit CUDAMemoryDeleter(std::uint64_t pool_id) : pool_id_(pool_id) {}
  void operator()(void *ptr) { CUDAMemoryPool::free(pool_id_, ptr); }
private:
  std::uint64_t pool_id_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CUDA_MEMORY_POOL_H_
