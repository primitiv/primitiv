#include <parameter_tuner.h>

#include <common.h>

using namespace primitiv;

class SetIdentityTuner : public devices::OpenCLParameterTuner {
public:
  SetIdentityTuner(std::size_t platform_id, std::size_t device_id)
  : devices::OpenCLParameterTuner(
      "set_identity_kernel", {"GROUP_SIZE"},
      platform_id, device_id, 5000, 2000) {}

  void iter_function() override {
    dev_->identity(200);
    dev_->identity(256);
    dev_->identity(300);
  }
};

PRIMITIV_TUNER_OPENCL_MAIN(SetIdentityTuner)
