#ifndef PRIMITIV_CORE_MEMORY_POOL_H_
#define PRIMITIV_CORE_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <primitiv/core/mixins/identifiable.h>

namespace primitiv {

/**
 * Memory manager on the device specified by allocator/deleter functors.
 */
class MemoryPool : public mixins::Identifiable<MemoryPool> {
  /**
   * Custom deleter class for MemoryPool.
   */
  class Deleter {
    std::uint64_t pool_id_;
  public:
    explicit Deleter(std::uint64_t pool_id) : pool_id_(pool_id) {}

    void operator()(void *ptr) {
      try {
        MemoryPool::get_object(pool_id_).free(ptr);
      } catch (const primitiv::Error&) {
        // Memory pool already has gone and the pointer is already deleted by
        // the memory pool.
      }
    }
  };

  std::function<void *(std::size_t)> allocator_;
  std::function<void(void *)> deleter_;
  std::vector<std::vector<void *>> reserved_;
  std::unordered_map<void *, std::uint32_t> supplied_;

public:
  /**
   * Creates a memory pool.
   * @param allocator Functor to allocate new memories.
   * @param deleter Functor to delete allocated memories.
   */
  explicit MemoryPool(
      std::function<void *(std::size_t)> allocator,
      std::function<void(void *)> deleter);

  ~MemoryPool();

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
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_MEMORY_POOL_H_
