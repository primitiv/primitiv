#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>

namespace primitiv {
namespace devices {

std::uint32_t OpenCL::num_platforms() {
  return ::get_all_platforms().size();
}

std::uint32_t OpenCL::num_devices(std::uint32_t platform_id) {
  return ::get_all_devices(platform_id).size();
}

void OpenCL::assert_support(
    std::uint32_t platform_id, std::uint32_t device_id) {
  const cl::Device dev = ::get_device(platform_id, device_id);

  // Checks whether the device is globally available.
  if (!dev.getInfo<CL_DEVICE_AVAILABLE>()) {
    PRIMITIV_THROW_ERROR(
        "OpenCL Device " << device_id << " on the platform " << platform_id
        << " is not available (CL_DEVICE_AVAILABLE == false).");
  }

  // Checks other minimum requirements.
#define CHECK_REQUIREMENT(name, value) \
  { \
    const auto actual = dev.getInfo<name>(); \
    if (actual < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << ", " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  }
#define CHECK_REQUIREMENT_VECTOR(name, index, value) \
  { \
    const auto actual = dev.getInfo<name>()[index]; \
    if (actual < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << "[" << #index << "], " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  } \

  CHECK_REQUIREMENT(CL_DEVICE_GLOBAL_MEM_SIZE, 1ull * (1ull << 30));
  CHECK_REQUIREMENT(CL_DEVICE_LOCAL_MEM_SIZE, 16ull * (1ull << 10));
  CHECK_REQUIREMENT(CL_DEVICE_MAX_WORK_GROUP_SIZE, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 1, 16);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 2, 1);
  // NOTE(odashi): OpenCL does not support explicit grid sizes.

#undef CHECK_REQUIREMENT
#undef CHECK_REQUIREMENT_VECTOR
}

void OpenCL::initialize() {
  assert_support(pf_id_, dev_id_);
  state_.reset(new OpenCLInternalState(pf_id_, dev_id_, rng_seed_));
}

OpenCL::OpenCL(std::uint32_t platform_id, std::uint32_t device_id)
: pf_id_(platform_id)
, dev_id_(device_id)
, rng_seed_(std::random_device()()) {
  initialize();
}

OpenCL::OpenCL(
    std::uint32_t platform_id, std::uint32_t device_id, std::uint32_t rng_seed)
: pf_id_(platform_id)
, dev_id_(device_id)
, rng_seed_(rng_seed) {
  initialize();
}

OpenCL::~OpenCL() {
  // Nothing to do for now.
}

}  // namespace devices
}  // namespace primitiv
