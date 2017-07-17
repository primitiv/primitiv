#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>

namespace primitiv {

class DeviceTest : public testing::Test {};

TEST_F(DeviceTest, CheckDefaultDevice) {
  EXPECT_THROW(Device::get_default_device(), Error);
  {
    CPUDevice dev;
    Device::set_default_device(dev);
    EXPECT_EQ(&dev, &Device::get_default_device());
  }
  EXPECT_THROW(Device::get_default_device(), Error);
  {
    CPUDevice dev1;
    Device::set_default_device(dev1);
    EXPECT_EQ(&dev1, &Device::get_default_device());
    CPUDevice dev2;
    Device::set_default_device(dev2);
    EXPECT_EQ(&dev2, &Device::get_default_device());
  }
  EXPECT_THROW(Device::get_default_device(), Error);

}

}  // namespace primitiv
