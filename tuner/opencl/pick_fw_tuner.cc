#include <parameter_tuner.h>
#include <common.h>

using namespace primitiv;

class PickFwTuner : public devices::OpenCLParameterTuner {
public:
  PickFwTuner(std::size_t platform_id, std::size_t device_id)
  : devices::OpenCLParameterTuner(
      "pick_fw_kernel", {"GROUP_SIZE"},
      platform_id, device_id, 5000, 500) {}

  void initialize() override {
    Shape s1({3, 256, 256});
    Shape s2({256, 3, 256});
    Shape s3({256, 256, 3});
    initializers::Uniform initializer(-1, 1);
    t1_ = dev_->new_tensor_by_constant(s1, 0);
    t2_ = dev_->new_tensor_by_constant(s2, 0);
    t3_ = dev_->new_tensor_by_constant(s3, 0);
    initializer.apply(t1_);
    initializer.apply(t2_);
    initializer.apply(t3_);
  }

  void iter_function() override {
    dev_->pick_fw(t1_, {2, 1, 0}, 0);
    dev_->pick_fw(t2_, {2, 1, 0}, 1);
    dev_->pick_fw(t3_, {2, 1, 0}, 2);
  }

private:
  Tensor t1_;
  Tensor t2_;
  Tensor t3_;
};

PRIMITIV_TUNER_OPENCL_MAIN(PickFwTuner)
