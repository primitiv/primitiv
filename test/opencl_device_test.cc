#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/opencl_device.h>

namespace primitiv {

class OpenCLDeviceTest : public testing::Test {};

TEST_F(OpenCLDeviceTest, CheckDeviceType) {
  devices::OpenCL dev(0, 0);
  EXPECT_EQ(Device::DEVICE_TYPE_OPENCL, dev.type());
}

//TODO(odashi): Add tests for OpenCL device.

}  // namespace primitiv
