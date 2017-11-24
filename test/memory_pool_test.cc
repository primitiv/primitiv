#include <config.h>

#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include <primitiv/memory_pool.h>
#include <primitiv/error.h>
#include <primitiv/shape.h>

namespace primitiv {

class MemoryPoolTest : public testing::Test {
protected:
  static void *allocator(std::size_t size) {
    return new char[size];
  }

  static void deleter(void *ptr) {
    delete[] static_cast<char *>(ptr);
  }
};

// This test should be placed on the top.
TEST_F(MemoryPoolTest, CheckNew) {
  {
    MemoryPool pool(allocator, deleter);
    EXPECT_EQ(0, pool.id());
  }  // pool is destroyed at the end of scope.
  SUCCEED();
}

TEST_F(MemoryPoolTest, CheckPoolIDs) {
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

TEST_F(MemoryPoolTest, CheckAllocate) {
  std::uint32_t allocated = 0;
  std::uint32_t deleted = 0;

  auto local_allocator = [&allocated](std::size_t size) {
    ++allocated;
    return allocator(size);
  };

  auto local_deleter = [&deleted](void *ptr) {
    ++deleted;
    deleter(ptr);
  };

  {
    MemoryPool pool(local_allocator, local_deleter);
    void *p1, *p2, *p3, *p4;
    {
      // Allocates new pointers.
      const auto sp1 = pool.allocate(1llu);
      const auto sp2 = pool.allocate(1llu << 8);
      const auto sp3 = pool.allocate(1llu << 16);
      const auto sp4 = pool.allocate(1llu << 24);

      EXPECT_EQ(4u, allocated);
      EXPECT_EQ(0u, deleted);

      p1 = sp1.get();
      p2 = sp2.get();
      p3 = sp3.get();
      p4 = sp4.get();
    }

    EXPECT_EQ(4u, allocated);
    EXPECT_EQ(0u, deleted);

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

      EXPECT_EQ(4u, allocated);
      EXPECT_EQ(0u, deleted);

      // Allocates other pointers.
      const auto sp11 = pool.allocate(1llu);
      const auto sp22 = pool.allocate(1llu << 8);
      const auto sp33 = pool.allocate(1llu << 16);
      const auto sp44 = pool.allocate(1llu << 24);
      EXPECT_NE(p1, sp11.get());
      EXPECT_NE(p2, sp22.get());
      EXPECT_NE(p3, sp33.get());
      EXPECT_NE(p4, sp44.get());

      EXPECT_EQ(8u, allocated);
      EXPECT_EQ(0u, deleted);
    }

    EXPECT_EQ(8u, allocated);
    EXPECT_EQ(0u, deleted);
  }

  EXPECT_EQ(8u, allocated);
  EXPECT_EQ(8u, deleted);
}

TEST_F(MemoryPoolTest, CheckAllocateMultithread) {
  std::uint32_t allocated = 0;

  auto local_allocator = [&allocated](std::size_t size) {
    ++allocated;
    return allocator(size);
  };

  MemoryPool pool(local_allocator, deleter);

  auto thread_proc = [&pool] {
    const auto p1 = pool.allocate(1llu);
    const auto p2 = pool.allocate(1llu << 8);
    const auto p3 = pool.allocate(1llu << 16);
    const auto p4 = pool.allocate(1llu << 24);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  };

  std::thread th1(thread_proc);
  std::thread th2(thread_proc);

  th1.join();
  th2.join();

  EXPECT_EQ(8u, allocated);
}

}  // namespace primitiv
