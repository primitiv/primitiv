#include <primitiv/config.h>

#include <iostream>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: OpenCL" << std::endl;

  std::cerr << "  Platform ID: " << pf_id_ << std::endl;
  std::cerr << "  Device ID: " << dev_id_ << std::endl;
  std::cerr << "    Vendor ............ "
            << state_->device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
  std::cerr << "    Name .............. "
            << state_->device.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cerr << "    Global memory ..... "
            << state_->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
  std::cerr << "    Local memory ...... "
            << state_->device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
  std::cerr << "    Work group size ... "
            << state_->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  std::cerr << "    Work item size .... ";
  const auto sizes = state_->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << sizes[i];
  }
  std::cerr << std::endl;
}

}  // namespace devices
}  // namespace primitiv
