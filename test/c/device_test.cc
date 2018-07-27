#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/c/devices/naive/device.h>

namespace primitiv {
namespace c {

class CDeviceTest : public testing::Test {};

TEST_F(CDeviceTest, CheckDefault) {
  ::primitivDevice_t *device;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetDefaultDevice(&device));
  {
    ::primitivDevice_t *dev1;
    ASSERT_EQ(PRIMITIV_C_OK,
              ::primitivCreateNaiveDevice(&dev1));
    ::primitivSetDefaultDevice(dev1);
    ::primitivGetDefaultDevice(&device);
    EXPECT_EQ(dev1, device);
    {
      ::primitivDevice_t *dev2;
    ASSERT_EQ(PRIMITIV_C_OK,
      ::primitivCreateNaiveDevice(&dev2));
      ::primitivSetDefaultDevice(dev2);
      ::primitivGetDefaultDevice(&device);
      EXPECT_EQ(dev2, device);
      ::primitivDeleteDevice(dev2);
    }
    EXPECT_EQ(PRIMITIV_C_ERROR,
              ::primitivGetDefaultDevice(&device));
    ::primitivDevice_t *dev3;
    ASSERT_EQ(PRIMITIV_C_OK,
    ::primitivCreateNaiveDevice(&dev3));
    ::primitivSetDefaultDevice(dev3);
    ::primitivGetDefaultDevice(&device);
    EXPECT_EQ(dev3, device);
    ::primitivDeleteDevice(dev1);
    ::primitivDeleteDevice(dev3);
  }
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetDefaultDevice(&device));
}

}  // namespace c
}  // namespace primitiv
