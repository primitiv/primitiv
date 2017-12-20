#include <primitiv/config.h>

#include <iostream>
#include <vector>

#include <primitiv/error.h>
#include <test_utils.h>

#include <primitiv/naive_device.h>
using primitiv::devices::Naive;

#ifdef PRIMITIV_USE_EIGEN
#include <primitiv/eigen_device.h>
using primitiv::devices::Eigen;
#endif  // PRIMITIV_USE_EIGEN

#ifdef PRIMITIV_USE_CUDA
#include <primitiv/cuda_device.h>
using primitiv::devices::CUDA;
#endif  // PRIMITIV_USE_CUDA

#ifdef PRIMITIV_USE_OPENCL
#include <primitiv/opencl_device.h>
using primitiv::devices::OpenCL;
#endif  // PRIMITIV_USE_OPENCL

namespace {

void add_device(std::vector<primitiv::Device *> &devs, primitiv::Device *dev) {
  devs.emplace_back(dev);
  dev->dump_description();
}

}  // namespace

namespace test_utils {

void add_available_devices(std::vector<primitiv::Device *> &devs) {
  add_available_naive_devices(devs);
  add_available_eigen_devices(devs);
  add_available_cuda_devices(devs);
  add_available_opencl_devices(devs);
}

void add_available_naive_devices(std::vector<primitiv::Device *> &devs) {
  // We can always add Naive devices.
  ::add_device(devs, new Naive());
  ::add_device(devs, new Naive());
}

void add_available_eigen_devices(std::vector<primitiv::Device *> &devs) {
#ifdef PRIMITIV_USE_EIGEN
  ::add_device(devs, new Eigen());
  ::add_device(devs, new Eigen());
#endif  // PRIMITIV_USE_EIGEN
}

void add_available_cuda_devices(std::vector<primitiv::Device *> &devs) {
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
      << " CUDA device(s) are not supported." << std::endl;
  }
  if (num_avail_devs == 0) {
    std::cerr << "No available CUDA devices." << std::endl;
  }
#endif  // PRIMITIV_USE_CUDA
}

void add_available_opencl_devices(std::vector<primitiv::Device *> &devs) {
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
