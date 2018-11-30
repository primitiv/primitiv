#include <parameter_tuner.h>
#include <common.h>

using namespace primitiv;

class FlipFwTuner : public devices::OpenCLParameterTuner {
public:
  FlipFwTuner(std::size_t platform_id, std::size_t device_id)
  : devices::OpenCLParameterTuner(
      "flip_fw_kernel", {"GROUP_SIZE_X", "GROUP_SIZE_Y"},
      platform_id, device_id, 5000, 500) {}

  void initialize() override {
    Shape s1({256, 256, 3});
    Shape s2({512, 128, 3});
    Shape s3({128, 512, 3});
    initializers::Uniform initializer(-1, 1);
    t1_ = dev_->new_tensor_by_constant(s1, 0);
    t2_ = dev_->new_tensor_by_constant(s2, 0);
    t3_ = dev_->new_tensor_by_constant(s3, 0);
    initializer.apply(t1_);
    initializer.apply(t2_);
    initializer.apply(t3_);
  }

  void iter_function() override {
    dev_->flip_fw(t1_, 0);
    dev_->flip_fw(t2_, 0);
    dev_->flip_fw(t3_, 0);
    dev_->flip_fw(t1_, 1);
    dev_->flip_fw(t2_, 1);
    dev_->flip_fw(t3_, 1);
    dev_->flip_fw(t1_, 2);
    dev_->flip_fw(t2_, 2);
    dev_->flip_fw(t3_, 2);
  }

private:
  Tensor t1_;
  Tensor t2_;
  Tensor t3_;
};

PRIMITIV_TUNER_OPENCL_MAIN(FlipFwTuner)
