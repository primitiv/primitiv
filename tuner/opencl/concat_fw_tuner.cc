#include <parameter_tuner.h>
#include <common.h>

using namespace primitiv;

class ConcatFwTuner : public devices::OpenCLParameterTuner {
public:
  ConcatFwTuner(std::size_t platform_id, std::size_t device_id)
  : devices::OpenCLParameterTuner(
      "concat_fw_kernel", {"GROUP_SIZE"},
      platform_id, device_id, 5000, 500) {}

  void initialize() override {
    Shape s1({128, 128, 3});
    Shape s2({128, 3, 128});
    Shape s3({3, 128, 128});
    initializers::Uniform initializer(-1, 1);
    t11_ = dev_->new_tensor_by_constant(s1, 0);
    t12_ = dev_->new_tensor_by_constant(s1, 0);
    t13_ = dev_->new_tensor_by_constant(s1, 0);
    t21_ = dev_->new_tensor_by_constant(s2, 0);
    t22_ = dev_->new_tensor_by_constant(s2, 0);
    t23_ = dev_->new_tensor_by_constant(s2, 0);
    t31_ = dev_->new_tensor_by_constant(s3, 0);
    t32_ = dev_->new_tensor_by_constant(s3, 0);
    t33_ = dev_->new_tensor_by_constant(s3, 0);
    initializer.apply(t11_);
    initializer.apply(t12_);
    initializer.apply(t13_);
    initializer.apply(t21_);
    initializer.apply(t22_);
    initializer.apply(t23_);
    initializer.apply(t31_);
    initializer.apply(t32_);
    initializer.apply(t33_);
  }

  void iter_function() override {
    dev_->concat_fw({&t11_, &t12_, &t13_}, 0);
    dev_->concat_fw({&t11_, &t12_, &t13_}, 1);
    dev_->concat_fw({&t11_, &t12_, &t13_}, 2);
    dev_->concat_fw({&t21_, &t22_, &t23_}, 0);
    dev_->concat_fw({&t21_, &t22_, &t23_}, 1);
    dev_->concat_fw({&t21_, &t22_, &t23_}, 2);
    dev_->concat_fw({&t31_, &t32_, &t33_}, 0);
    dev_->concat_fw({&t31_, &t32_, &t33_}, 1);
    dev_->concat_fw({&t31_, &t32_, &t33_}, 2);
  }

private:
  Tensor t11_;
  Tensor t12_;
  Tensor t13_;
  Tensor t21_;
  Tensor t22_;
  Tensor t23_;
  Tensor t31_;
  Tensor t32_;
  Tensor t33_;
};

PRIMITIV_TUNER_OPENCL_MAIN(ConcatFwTuner)
