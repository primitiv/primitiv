#include <primitiv/config.h>

#include <iostream>
#include <vector>

#include <primitiv/core/error.h>
#include <test_utils.h>

#include <primitiv/devices/naive/device.h>
using primitiv::devices::Naive;

#ifdef PRIMITIV_USE_EIGEN
#include <primitiv/devices/eigen/device.h>
using primitiv::devices::Eigen;
#endif  // PRIMITIV_USE_EIGEN

#ifdef PRIMITIV_USE_CUDA
#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda16/device.h>
using primitiv::devices::CUDA;
using primitiv::devices::CUDA16;
#endif  // PRIMITIV_USE_CUDA

#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/devices/opencl/device.h>
using primitiv::devices::OpenCL;
#endif  // PRIMITIV_USE_OPENCL

#define MAYBE_USED(x) static_cast<void>(x)

namespace {

void add_device(std::vector<primitiv::Device *> &devs, primitiv::Device *dev) {
  devs.emplace_back(dev);
  dev->dump_description();
}

}  // namespace

namespace test_utils {

std::uint32_t get_default_ulps(const primitiv::Device &dev) {
  switch (dev.type()) {
    case primitiv::DeviceType::CUDA16:
      // NOTE(odashi):
      // Returns the half of the difference of the resolution between
      // float (23 bits) and half (10 bits).
      return 1 << (13 - 1);
    default:
      // NOTE(odashi):
      // 4-ULPs test is identical with the ASSERT(EXPECT)_FLOAT_EQ directives in
      // the Google Test.
      return 4;
  }
}

void add_available_devices(std::vector<primitiv::Device *> &devs) {
  add_available_naive_devices(devs);
  add_available_eigen_devices(devs);
  add_available_cuda_devices(devs);
  add_available_cuda16_devices(devs);
  add_available_opencl_devices(devs);
}

void add_available_naive_devices(std::vector<primitiv::Device *> &devs) {
  // We can always add Naive devices.
  ::add_device(devs, new Naive());
  ::add_device(devs, new Naive());
}

void add_available_eigen_devices(std::vector<primitiv::Device *> &devs) {
  MAYBE_USED(devs);
#ifdef PRIMITIV_USE_EIGEN
  ::add_device(devs, new Eigen());
  ::add_device(devs, new Eigen());
#endif  // PRIMITIV_USE_EIGEN
}

void add_available_cuda_devices(std::vector<primitiv::Device *> &devs) {
  MAYBE_USED(devs);
#ifdef PRIMITIV_USE_CUDA
  const std::uint32_t num_devs = CUDA::num_devices();
  std::uint32_t num_avail_devs = 0;
  for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
    if (CUDA::check_support(dev_id)) {
      ++num_avail_devs;
      ::add_device(devs, new CUDA(dev_id));
      if (num_avail_devs == 1) {
        // Add another device object on the first device.
        ::add_device(devs, new CUDA(dev_id));
      }
    }
  }
  if (num_avail_devs != num_devs) {
    std::cerr << (num_devs - num_avail_devs)
      << " device(s) are not supported by primitiv::devices::CUDA."
      << std::endl;
  }
  if (num_avail_devs == 0) {
    std::cerr << "No available CUDA devices." << std::endl;
  }
#endif  // PRIMITIV_USE_CUDA
}

void add_available_cuda16_devices(std::vector<primitiv::Device *> &devs) {
  MAYBE_USED(devs);
#ifdef PRIMITIV_USE_CUDA
  const std::uint32_t num_devs = CUDA16::num_devices();
  std::uint32_t num_avail_devs = 0;
  for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
    if (CUDA16::check_support(dev_id)) {
      ++num_avail_devs;
      ::add_device(devs, new CUDA16(dev_id));
      if (num_avail_devs == 1) {
        // Add another device object on the first device.
        ::add_device(devs, new CUDA16(dev_id));
      }
    }
  }
  if (num_avail_devs != num_devs) {
    std::cerr << (num_devs - num_avail_devs)
      << " device(s) are not supported by primitiv::devices::CUDA16."
      << std::endl;
  }
  if (num_avail_devs == 0) {
    std::cerr << "No available CUDA16 devices." << std::endl;
  }
#endif  // PRIMITIV_USE_CUDA
}

void add_available_opencl_devices(std::vector<primitiv::Device *> &devs) {
  MAYBE_USED(devs);
#ifdef PRIMITIV_USE_OPENCL
  const std::uint32_t num_pfs = OpenCL::num_platforms();
  if (num_pfs > 0) {
    for (std::uint32_t pf_id = 0; pf_id < num_pfs; ++pf_id) {
      const std::uint32_t num_devs = OpenCL::num_devices(pf_id);
      std::uint32_t num_avail_devs = 0;
      for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
        if (OpenCL::check_support(pf_id, dev_id)) {
          ++num_avail_devs;
          ::add_device(devs, new OpenCL(pf_id, dev_id));
          if (num_avail_devs == 1) {
            // Add another device object on the device 0.
            ::add_device(devs, new OpenCL(pf_id, dev_id));
          }
        }
      }
      if (num_avail_devs != num_devs) {
        std::cerr << (num_devs - num_avail_devs)
          << " OpenCL device(s) are not supported." << std::endl;
      }
      if (num_avail_devs == 0) {
        std::cerr << "No available OpenCL devices." << std::endl;
      }
    }
  } else {
    std::cerr << "No OpenCL platforms are installed." << std::endl;
  }
#endif  // PRIMITIV_USE_OPENCL
}

}  // namespace test_utils

#undef MAYBE_USED
