#include <config.h>

#include <stdexcept>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>

namespace primitiv {

class CPUDeviceTest : public testing::Test {};

TEST_F(CPUDeviceTest, CheckAllocFree) {
  {
    CPUDevice dev;
    EXPECT_EQ(0u, dev.num_blocks());

    // free(nullptr) is always OK.
    EXPECT_NO_THROW(dev.free(nullptr));
  }
  {
    CPUDevice dev;
    void *ptr1 = dev.allocate(1); // 1B
    EXPECT_EQ(1u, dev.num_blocks());
    void *ptr2 = dev.allocate(2 << 10); // 1kB
    EXPECT_EQ(2u, dev.num_blocks());
    void *ptr3 = dev.allocate(2 << 20); // 1MB
    EXPECT_EQ(3u, dev.num_blocks());
    dev.free(ptr1);
    EXPECT_EQ(2u, dev.num_blocks());
    dev.free(ptr2);
    EXPECT_EQ(1u, dev.num_blocks());
    dev.free(ptr3);
    EXPECT_EQ(0u, dev.num_blocks());
  }
}

TEST_F(CPUDeviceTest, CheckInvalidAlloc) {
  CPUDevice dev;
  EXPECT_THROW(dev.allocate(0), std::runtime_error);
}

TEST_F(CPUDeviceTest, CheckInvalidFree) {
  CPUDevice dev;
  void *ptr = dev.allocate(1);
  EXPECT_THROW(dev.free(reinterpret_cast<void *>(1)), std::runtime_error);
  EXPECT_THROW(dev.free(static_cast<char *>(ptr) + 1), std::runtime_error);
  dev.free(ptr);
  EXPECT_THROW(dev.free(ptr), std::runtime_error);
}

TEST_F(CPUDeviceTest, CheckMemoryLeak) {
  EXPECT_DEATH({
      CPUDevice dev;
      dev.allocate(1);
  }, "");
}

}  // namespace primitiv
