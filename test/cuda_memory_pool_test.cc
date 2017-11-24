#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/memory_pool.h>
#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>

namespace primitiv {

class CUDAMemoryPoolTest : public testing::Test {
protected:
  static void *allocator(std::size_t size) {
    void *ptr;
    CUDA_CALL(::cudaSetDevice(0));
    CUDA_CALL(::cudaMalloc(&ptr, size));
    return ptr;
  }

  static void deleter(void *ptr) {
    CUDA_CALL(::cudaFree(ptr));
  }
};

TEST_F(CUDAMemoryPoolTest, CheckInvalidAllocate) {
  MemoryPool pool(allocator, deleter);

  // Available maximum size of the memory: 2^63 bytes.
  EXPECT_THROW(pool.allocate((1llu << 63) + 1), Error);

  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, 0));

  // Calculates the half or more size of the whole memory.
  std::size_t size = 1llu;
  while (size < prop.totalGlobalMem >> 1) size <<= 1;

  // rvalue shared pointers immediately releases the raw pointers, so that the
  // out-of-memory error does not occur.
  ASSERT_NO_THROW(pool.allocate(size));
  ASSERT_NO_THROW(pool.allocate(size));
  ASSERT_NO_THROW(pool.allocate(size));
  ASSERT_NO_THROW(pool.allocate(size));
  ASSERT_NO_THROW(pool.allocate(size));

  // Binds large memory to an lvalue.
  std::shared_ptr<void> sp;
  ASSERT_NO_THROW(sp = pool.allocate(size));
  EXPECT_THROW(pool.allocate(size), Error);  // Out of memory
}

TEST_F(CUDAMemoryPoolTest, CheckReleaseReservedBlocks) {
  MemoryPool pool(allocator, deleter);

  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, 0));
  std::cerr << "Total size: " << prop.totalGlobalMem << std::endl;

  // Calculates chunk sizes and number of chunks.
  std::size_t size = 1llu;
  while (size < prop.totalGlobalMem >> 3) size <<= 1;
  std::uint32_t n = 0;
  while (n * size < prop.totalGlobalMem) ++n;
  std::cerr << "Chunk size: " << size << std::endl;
  std::cerr << "#Chunks: " << n << std::endl;
  ASSERT_LE(4u, n);

  // Reserves n-4 chunks.
  std::vector<std::shared_ptr<void>> reserved;
  for (std::uint32_t i = 0; i < n - 4; ++i) {
    EXPECT_NO_THROW(reserved.emplace_back(pool.allocate(size)));
  }

  std::shared_ptr<void> p1;
  EXPECT_NO_THROW(p1 = pool.allocate(size));  // 1/4 of all
  {
    std::shared_ptr<void> p2;
    EXPECT_NO_THROW(p2 = pool.allocate(size));  // 2/4 of all
    EXPECT_THROW(pool.allocate(size << 1), Error);  // 4/4 of all (OOM)
  }
  {
    std::shared_ptr<void> p2;
    EXPECT_NO_THROW(p2 = pool.allocate(size << 1)); // 3/4 of all
    EXPECT_THROW(pool.allocate(size), Error);  // 4/4 of all (OOM)
  }
  EXPECT_NO_THROW(pool.allocate(size));  // 2/4 of all
}

}  // namespace primitiv
