#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/opencl_device.h>

namespace primitiv {

class OpenCLDeviceTest : public testing::Test {};

TEST_F(OpenCLDeviceTest, CheckDeviceType) {
  const std::uint32_t num_pfs = devices::OpenCL::num_platforms();
  for (std::uint32_t pf_id = 0; pf_id < num_pfs; ++pf_id) {
    const std::uint32_t num_devs = devices::OpenCL::num_devices(pf_id);
    for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
      try {
        devices::OpenCL dev(pf_id, dev_id);
        EXPECT_EQ(Device::DeviceType::OPENCL, dev.type());
      } catch (Error e) {
        std::cout << e.what() << std::endl;
      }
    }
  }
}

//TODO(odashi): Add tests for OpenCL device.

}  // namespace primitiv
