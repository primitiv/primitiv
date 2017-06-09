#ifndef PRIMITIV_CUDA_MEMORY_POOL_H_
#define PRIMITIV_CUDA_MEMORY_POOL_H_

#include <unordered_map>
#include <utility>
#include <vector>

namespace primitiv {

/**
 * Memory manager on the CUDA devices.
 */
class CUDAMemoryPool {
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
   * @return Handle of the allocated memory.
   */
  void *allocate(unsigned size);
  
  /**
   * Disposes the memory.
   * @param ptr Handle of the memory to be disposed.
   */
  void free(void *ptr);

private:
  unsigned dev_id_;
  std::vector<std::vector<std::pair<void *, unsigned>>> reserved_;
  std::unordered_map<void *, unsigned> supplid_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CUDA_MEMORY_POOL_H_
