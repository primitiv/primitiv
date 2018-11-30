#include <parameter_tuner.h>
#include <common.h>

using namespace primitiv;

class PickBwTuner : public devices::OpenCLParameterTuner {
public:
  PickBwTuner(std::size_t platform_id, std::size_t device_id)
  : devices::OpenCLParameterTuner(
      "pick_bw_kernel", {"GROUP_SIZE"},
      platform_id, device_id, 5000, 500) {}

  void initialize() override {
    Shape sx1({3, 256, 256});
    Shape sx2({256, 3, 256});
    Shape sx3({256, 256, 3});
    Shape sy1({1, 256, 256}, 3);
    Shape sy2({256, 1, 256}, 3);
    Shape sy3({256, 256, 1}, 3);
    initializers::Uniform initializer(-1, 1);
    tx1_ = dev_->new_tensor_by_constant(sx1, 0);
    tx2_ = dev_->new_tensor_by_constant(sx2, 0);
    tx3_ = dev_->new_tensor_by_constant(sx3, 0);
    ty1_ = dev_->new_tensor_by_constant(sy1, 0);
    ty2_ = dev_->new_tensor_by_constant(sy2, 0);
    ty3_ = dev_->new_tensor_by_constant(sy3, 0);
    initializer.apply(tx1_);
    initializer.apply(tx2_);
    initializer.apply(tx3_);
    initializer.apply(ty1_);
    initializer.apply(ty2_);
    initializer.apply(ty3_);
  }

  void iter_function() override {
    dev_->pick_bw(ty1_, {2, 1, 0}, 0, tx1_);
    dev_->pick_bw(ty2_, {2, 1, 0}, 1, tx2_);
    dev_->pick_bw(ty3_, {2, 1, 0}, 2, tx3_);
  }

private:
  Tensor tx1_;
  Tensor tx2_;
  Tensor tx3_;
  Tensor ty1_;
  Tensor ty2_;
  Tensor ty3_;
};

PRIMITIV_TUNER_OPENCL_MAIN(PickBwTuner)
