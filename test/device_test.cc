#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/devices/naive/device.h>

namespace primitiv {

class DeviceTest : public testing::Test {};

TEST_F(DeviceTest, CheckDefault) {
  EXPECT_THROW(Device::get_default(), Error);
  {
    devices::Naive dev1;
    Device::set_default(dev1);
    EXPECT_EQ(&dev1, &Device::get_default());
    {
      devices::Naive dev2;
      Device::set_default(dev2);
      EXPECT_EQ(&dev2, &Device::get_default());
    }
    EXPECT_THROW(Device::get_default(), Error);
    devices::Naive dev3;
    Device::set_default(dev3);
    EXPECT_EQ(&dev3, &Device::get_default());
  }
  EXPECT_THROW(Device::get_default(), Error);
}

}  // namespace primitiv
