#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/core/memory_pool.h>
#include <primitiv/core/error.h>
#include <primitiv/core/shape.h>
#include <primitiv/devices/cuda/device.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {

class MemoryPoolTest_CUDA : public testing::Test {
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

// This test should be placed on the top.
TEST_F(MemoryPoolTest_CUDA, CheckNew) {
  {
    MemoryPool pool(allocator, deleter);
    EXPECT_EQ(0u, pool.id());
  }  // pool is destroyed at the end of scope.
  SUCCEED();
}

TEST_F(MemoryPoolTest_CUDA, CheckPoolIDs) {
  MemoryPool pool0(allocator, deleter);
  std::uint64_t base_id = pool0.id();

  MemoryPool pool1(allocator, deleter);
  EXPECT_EQ(base_id + 1, pool1.id());
  MemoryPool(allocator, deleter);
  MemoryPool(allocator, deleter);
  MemoryPool pool2(allocator, deleter);
  EXPECT_EQ(base_id + 4, pool2.id());
  {
    MemoryPool pool3(allocator, deleter);
    EXPECT_EQ(base_id + 5, pool3.id());
    MemoryPool(allocator, deleter);
    MemoryPool(allocator, deleter);
    MemoryPool pool4(allocator, deleter);
    EXPECT_EQ(base_id + 8, pool4.id());
  }
  MemoryPool pool5(allocator, deleter);
  EXPECT_EQ(base_id + 9, pool5.id());
}

TEST_F(MemoryPoolTest_CUDA, CheckEmptyAllocation) {
  MemoryPool pool(allocator, deleter);
  const auto sp1 = pool.allocate(0u);
  const auto sp2 = pool.allocate(0u);
  const auto sp3 = pool.allocate(0u);
  EXPECT_EQ(nullptr, sp1.get());
  EXPECT_EQ(nullptr, sp2.get());
  EXPECT_EQ(nullptr, sp3.get());
}

TEST_F(MemoryPoolTest_CUDA, CheckAllocate) {
  MemoryPool pool(allocator, deleter);
  void *p1, *p2, *p3, *p4;
  {
    // Allocates new pointers.
    const auto sp1 = pool.allocate(1llu);
    const auto sp2 = pool.allocate(1llu << 8);
    const auto sp3 = pool.allocate(1llu << 16);
    const auto sp4 = pool.allocate(1llu << 24);
    p1 = sp1.get();
    p2 = sp2.get();
    p3 = sp3.get();
    p4 = sp4.get();
  }
  // sp1-4 are released at the end of above scope, but the raw pointer is kept
  // in the pool object.
  {
    // Allocates existing pointers.
    const auto sp1 = pool.allocate(1llu);
    const auto sp2 = pool.allocate(1llu << 8);
    const auto sp3 = pool.allocate(1llu << 16);
    const auto sp4 = pool.allocate(1llu << 24);
    EXPECT_EQ(p1, sp1.get());
    EXPECT_EQ(p2, sp2.get());
    EXPECT_EQ(p3, sp3.get());
    EXPECT_EQ(p4, sp4.get());
    // Allocates other pointers.
    const auto sp11 = pool.allocate(1llu);
    const auto sp22 = pool.allocate(1llu << 8);
    const auto sp33 = pool.allocate(1llu << 16);
    const auto sp44 = pool.allocate(1llu << 24);
    EXPECT_NE(p1, sp11.get());
    EXPECT_NE(p2, sp22.get());
    EXPECT_NE(p3, sp33.get());
    EXPECT_NE(p4, sp44.get());
  }
}

TEST_F(MemoryPoolTest_CUDA, CheckInvalidAllocate) {
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

TEST_F(MemoryPoolTest_CUDA, CheckReleaseReservedBlocks) {
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
