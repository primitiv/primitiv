#include <config.h>

#include <iostream>
#include <vector>
#include <test_utils.h>

#include <primitiv/error.h>

#include <primitiv/naive_device.h>
using primitiv::devices::Naive;

#ifdef PRIMITIV_USE_CUDA
#include <primitiv/cuda_device.h>
using primitiv::devices::CUDA;
#endif  // PRIMITIV_USE_CUDA

#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/opencl_device.h>
using primitiv::devices::OpenCL;
#endif  // PRIMITIV_USE_OPENCL

namespace test_utils {

void add_available_devices(std::vector<primitiv::Device *> &devs) {
  // We can always add Naive devices.
  {
    devs.emplace_back(new Naive());
    devs.emplace_back(new Naive());
  }

#ifdef PRIMITIV_USE_CUDA
  {
    const std::uint32_t num_devs = CUDA::num_devices();
    std::uint32_t num_avail_devs = 0;
    for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
      if (CUDA::check_support(dev_id)) {
        ++num_avail_devs;
        devs.emplace_back(new CUDA(dev_id));
        if (num_avail_devs == 1) {
          // Add another device object on the first device.
          devs.emplace_back(new CUDA(dev_id));
        }
      }
    }
    if (num_avail_devs == 0) {
      std::cout << "No available CUDA devices are installed";
      if (num_devs > 0) {
        std::cout << " (but found " << num_devs << " devices)";
      }
      std::cout << "." << std::endl;
    }
  }
#endif  // PRIMITIV_USE_CUDA

#ifdef PRIMITIV_USE_OPENCL
  {
    const std::uint32_t num_pfs = OpenCL::num_platforms();
    if (num_pfs > 0) {
      for (std::uint32_t pf_id = 0; pf_id < num_pfs; ++pf_id) {
        const std::uint32_t num_devs = OpenCL::num_devices(pf_id);
        if (num_devs > 0) {
          for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
            OpenCL *ocl_dev;
            try {
              ocl_dev = new OpenCL(pf_id, dev_id);
            } catch (primitiv::Error e) {
              std::cout << e.what() << std::endl;
              continue;
            }
            devs.emplace_back(ocl_dev);
            if (dev_id == 0) {
              // Add another device object on the device 0.
              try {
                ocl_dev = new OpenCL(pf_id, dev_id);
              } catch (primitiv::Error e) {
                std::cout << e.what() << std::endl;
                continue;
              }
              devs.emplace_back(ocl_dev);
            }
          }
        } else {
          std::cout << "No OpenCL devices on the platform " << pf_id <<
            " are installed." << std::endl;
        }
      }
    } else {
      std::cout << "No OpenCL platforms are installed." << std::endl;
    }
  }
#endif  // PRIMITIV_USE_OPENCL

  // Dumps descriptions about all devices.
  for (primitiv::Device *dev : devs) {
    dev->dump_description();
  }
}

}  // namespace test_utils
