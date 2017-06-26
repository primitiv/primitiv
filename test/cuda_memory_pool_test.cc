#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/cuda_memory_pool.h>
#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>

namespace primitiv {

class CUDAMemoryPoolTest : public testing::Test {};

TEST_F(CUDAMemoryPoolTest, CheckNew) {
  {
    CUDAMemoryPool pool(0);
  }  // pool is destroyed at the end of scope.
  SUCCEED();
}

TEST_F(CUDAMemoryPoolTest, CheckInvalidNew) {
  // We might not have 12345678 GPUs.
  EXPECT_THROW(CUDAMemoryPool(12345678), Error);
}

TEST_F(CUDAMemoryPoolTest, CheckAllocate) {
  CUDAMemoryPool pool(0);
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

TEST_F(CUDAMemoryPoolTest, CheckInvalidAllocate) {
  CUDAMemoryPool pool(0);

  // Available maximum size of the memory: 2^63 bytes.
  EXPECT_THROW(pool.allocate((1llu << 63) + 1), Error);

  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, 0));
  std::uint64_t size = 1llu;
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

}  // namespace primitiv
