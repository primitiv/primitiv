#include <primitiv/config.h>

#include <mutex>
#include <thread>

#include <gtest/gtest.h>

#include <primitiv/core/spinlock.h>

namespace primitiv {

class SpinlockTest : public testing::Test {};

TEST_F(SpinlockTest, CheckBlock) {
  Spinlock sl;
  int x = 0, y = 0;

  auto proc = [&] {
    for (int i = 0; i < 1000000; ++i) {
      ++x;
      std::lock_guard<Spinlock> lock(sl);
      ++y;
    }
  };

  std::thread th1(proc);
  std::thread th2(proc);
  std::thread th3(proc);
  std::thread th4(proc);

  th1.join();
  th2.join();
  th3.join();
  th4.join();

  EXPECT_NE(4000000, x);
  EXPECT_EQ(4000000, y);
}

class RecursiveSpinlockTest : public testing::Test {};

TEST_F(RecursiveSpinlockTest, CheckBlock) {
  RecursiveSpinlock sl;
  int x = 0;

  auto proc = [&] {
    for (int i = 0; i < 1000000; ++i) {
      {
        std::lock_guard<RecursiveSpinlock> lock1(sl);
        {
          std::lock_guard<RecursiveSpinlock> lock2(sl);
          {
            std::lock_guard<RecursiveSpinlock> lock3(sl);
            ++x;
          }
        }
      }
    }
  };

  std::thread th1(proc);
  std::thread th2(proc);
  std::thread th3(proc);
  std::thread th4(proc);

  th1.join();
  th2.join();
  th3.join();
  th4.join();

  EXPECT_EQ(4000000, x);
}

}  // namespace primitiv
