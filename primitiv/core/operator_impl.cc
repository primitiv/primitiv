#include <primitiv/config.h>

#include <algorithm>

#include <primitiv/core/device.h>
#include <primitiv/core/error.h>
#include <primitiv/core/functions.h>
#include <primitiv/core/operator_impl.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/shape_ops.h>
#include <primitiv/core/string_utils.h>

#define UNUSED(x) static_cast<void>(x)

using std::vector;

namespace primitiv {
namespace operators {

/*
 * Constructors.
 */

Input::Input(const Shape &shape, const vector<float> &data, Device &device)
: shape_(shape)
, data_(data)
, device_(device) {
  if (data_.size() != shape_.size()) {
    PRIMITIV_THROW_ERROR(
        "Data sizes mismatched."
        << " operator: Input"
        << ", required: " << shape_.size() << " (" << shape_.to_string() << ")"
        << ", actual: " << data_.size());
  }
}

/*
 * Operator names.
 */

#define IMPL_NAME_0(cls) std::string cls::name() const { return #cls; }
#define IMPL_NAME_1(cls, k) \
  std::string cls::name() const { \
  return #cls "(" + string_utils::to_string(k) + ')'; \
}
#define IMPL_NAME_2(cls, k1, k2) \
  std::string cls::name() const { \
  return #cls "(" \
    + string_utils::to_string(k1) + ',' \
    + string_utils::to_string(k2) + ')'; \
}

IMPL_NAME_0(Input);
IMPL_NAME_0(Parameter);
IMPL_NAME_0(Copy);
IMPL_NAME_1(Constant, k_);
IMPL_NAME_1(Identity, size_);
IMPL_NAME_1(RandomBernoulli, p_);
IMPL_NAME_2(RandomUniform, lower_, upper_);
IMPL_NAME_2(RandomNormal, mean_, sd_);
IMPL_NAME_2(RandomLogNormal, mu_, beta_);
IMPL_NAME_1(Pick, dim_);

std::string Slice::name() const {
  return "Slice("
    + string_utils::to_string(dim_) + ','
    + string_utils::to_string(lower_) + ':'
    + string_utils::to_string(upper_) + ')';
}

IMPL_NAME_2(Split, dim_, n_);
IMPL_NAME_1(Concat, dim_);

std::string Reshape::name() const {
  return "Reshape(" + shape_.to_string() + ')';
}

IMPL_NAME_1(Max, dim_);
IMPL_NAME_1(Min, dim_);
IMPL_NAME_1(Sum, dim_);
IMPL_NAME_1(LogSumExp, dim_);
IMPL_NAME_2(Broadcast, dim_, size_);
IMPL_NAME_1(SoftmaxCrossEntropy, dim_);
IMPL_NAME_1(SparseSoftmaxCrossEntropy, dim_);
IMPL_NAME_0(StopGradient);
IMPL_NAME_0(Flatten);
IMPL_NAME_0(Positive);
IMPL_NAME_0(Negative);

IMPL_NAME_1(AddConst, k_);
IMPL_NAME_1(SubtractConstR, k_);
IMPL_NAME_1(SubtractConstL, k_);
IMPL_NAME_1(MultiplyConst, k_);
IMPL_NAME_1(DivideConstR, k_);
IMPL_NAME_1(DivideConstL, k_);
IMPL_NAME_1(PowConstR, k_);
IMPL_NAME_1(PowConstL, k_);
IMPL_NAME_1(PReLU, k_);
IMPL_NAME_1(ELU, k_);

IMPL_NAME_1(PowN, k_);

IMPL_NAME_0(AddScalar);
IMPL_NAME_0(SubtractScalarR);
IMPL_NAME_0(SubtractScalarL);
IMPL_NAME_0(MultiplyScalar);
IMPL_NAME_0(DivideScalarR);
IMPL_NAME_0(DivideScalarL);
IMPL_NAME_0(PowScalarR);
IMPL_NAME_0(PowScalarL);

IMPL_NAME_0(Add);
IMPL_NAME_0(Subtract);
IMPL_NAME_0(Multiply);
IMPL_NAME_0(Divide);
IMPL_NAME_0(Pow);

IMPL_NAME_0(Transpose);
IMPL_NAME_0(PermuteDims);
IMPL_NAME_0(MatrixMultiply);

IMPL_NAME_1(Flip, dim_);

IMPL_NAME_0(Abs);
IMPL_NAME_0(Sqrt);
IMPL_NAME_0(Exp);
IMPL_NAME_0(Log);
IMPL_NAME_0(Tanh);
IMPL_NAME_0(Sigmoid);
IMPL_NAME_0(Softplus);
IMPL_NAME_0(Sin);
IMPL_NAME_0(Cos);
IMPL_NAME_0(Tan);
IMPL_NAME_0(ReLU);
IMPL_NAME_0(LReLU);

IMPL_NAME_0(BatchPick);
std::string BatchSlice::name() const {
  return "BatchSlice("
    + string_utils::to_string(lower_) + ':'
    + string_utils::to_string(upper_) + ')';
}
IMPL_NAME_1(BatchSplit, n_);
IMPL_NAME_0(BatchConcat);
IMPL_NAME_0(BatchSum);

std::string Convolution2D::name() const {
  return "Convolution2D("
    + string_utils::to_string(padding0_) + ','
    + string_utils::to_string(padding1_) + ','
    + string_utils::to_string(stride0_) + ','
    + string_utils::to_string(stride1_) + ','
    + string_utils::to_string(dilation0_) + ','
    + string_utils::to_string(dilation1_) + ')';
}

std::string MaxPooling2D::name() const {
  return "MaxPooling2D("
    + string_utils::to_string(window0_) + ','
    + string_utils::to_string(window1_) + ','
    + string_utils::to_string(padding0_) + ','
    + string_utils::to_string(padding1_) + ','
    + string_utils::to_string(stride0_) + ','
    + string_utils::to_string(stride1_) + ')';
}

#undef IMPL_NAME_0
#undef IMPL_NAME_1
#undef IMPL_NAME_2

/*
 * Shape forwarding operations.
 */

#define FWD_SHAPE(name) \
  void name::forward_shape( \
      const vector<const Shape *> &x, \
      const vector<Shape *> &y) const

#define FWD_SHAPE_UNARY(name) \
  FWD_SHAPE(name) { *y[0] = *x[0]; }

#define FWD_SHAPE_SCALAR(name) \
  FWD_SHAPE(name) { *y[0] = shape_ops::scalar_op(*x[0], *x[1]); }

#define FWD_SHAPE_ELEMENTWISE(name) \
  FWD_SHAPE(name) { *y[0] = shape_ops::elementwise(*x[0], *x[1]); }

FWD_SHAPE(Input) { UNUSED(x); *y[0] = shape_; }
FWD_SHAPE(Parameter) { UNUSED(x); *y[0] = param_.shape(); }
FWD_SHAPE(Copy) { *y[0] = *x[0]; }
FWD_SHAPE(Constant) { UNUSED(x); *y[0] = shape_; }
FWD_SHAPE(Identity) { UNUSED(x); *y[0] = Shape({size_, size_}); }
FWD_SHAPE(RandomBernoulli) { UNUSED(x); *y[0] = shape_; }
FWD_SHAPE(RandomUniform) { UNUSED(x); *y[0] = shape_; }
FWD_SHAPE(RandomNormal) { UNUSED(x); *y[0] = shape_; }
FWD_SHAPE(RandomLogNormal) { UNUSED(x); *y[0] = shape_; }
FWD_SHAPE(Pick) { *y[0] = shape_ops::pick(*x[0], ids_, dim_); }
FWD_SHAPE(Slice) { *y[0] = shape_ops::slice(*x[0], dim_, lower_, upper_); }
FWD_SHAPE(Split) {
  if (n_ == 0) {
    PRIMITIV_THROW_ERROR("Invalid number of partitions: " << n_);
  }
  Shape xs = *x[0];
  const std::uint32_t total = xs[dim_];
  const std::uint32_t span = total / n_;
  if (span * n_ != total) {
    PRIMITIV_THROW_ERROR(
        "Could not split the axis " << dim_ << " with size "
        << total << " into " << n_ << " partitions.");
  }
  xs.update_dim(dim_, span);
  for (std::uint32_t i = 0; i < n_; ++i) {
    *y[i] = xs;
  }
}
FWD_SHAPE(Concat) { *y[0] = shape_ops::concat(x, dim_); }
FWD_SHAPE(Reshape) { *y[0] = shape_ops::reshape(*x[0], shape_); }
FWD_SHAPE(Flatten) { *y[0] = shape_ops::flatten(*x[0]); }
FWD_SHAPE_UNARY(Positive);
FWD_SHAPE_UNARY(Negative);
FWD_SHAPE_UNARY(AddConst);
FWD_SHAPE_UNARY(SubtractConstR);
FWD_SHAPE_UNARY(SubtractConstL);
FWD_SHAPE_UNARY(MultiplyConst);
FWD_SHAPE_UNARY(DivideConstR);
FWD_SHAPE_UNARY(DivideConstL);
FWD_SHAPE_UNARY(PowConstR);
FWD_SHAPE_UNARY(PowConstL);
FWD_SHAPE_UNARY(Abs);
FWD_SHAPE_UNARY(Sqrt);
FWD_SHAPE_UNARY(Exp);
FWD_SHAPE_UNARY(Log);
FWD_SHAPE_UNARY(Tanh);
FWD_SHAPE_UNARY(Sigmoid);
FWD_SHAPE_UNARY(Softplus);
FWD_SHAPE_UNARY(Sin);
FWD_SHAPE_UNARY(Cos);
FWD_SHAPE_UNARY(Tan);
FWD_SHAPE_UNARY(ReLU);
FWD_SHAPE_UNARY(LReLU);
FWD_SHAPE_UNARY(PReLU);
FWD_SHAPE_UNARY(ELU);
FWD_SHAPE_UNARY(PowN);
FWD_SHAPE_UNARY(Flip);
FWD_SHAPE_SCALAR(AddScalar);
FWD_SHAPE_SCALAR(SubtractScalarR);
FWD_SHAPE_SCALAR(SubtractScalarL);
FWD_SHAPE_SCALAR(MultiplyScalar);
FWD_SHAPE_SCALAR(DivideScalarR);
FWD_SHAPE_SCALAR(DivideScalarL);
FWD_SHAPE_SCALAR(PowScalarR);
FWD_SHAPE_SCALAR(PowScalarL);
FWD_SHAPE_ELEMENTWISE(Add);
FWD_SHAPE_ELEMENTWISE(Subtract);
FWD_SHAPE_ELEMENTWISE(Multiply);
FWD_SHAPE_ELEMENTWISE(Divide);
FWD_SHAPE_ELEMENTWISE(Pow);
FWD_SHAPE(Transpose) { *y[0] = shape_ops::transpose(*x[0]); }
FWD_SHAPE(PermuteDims) { *y[0] = shape_ops::permute_dims(*x[0], perm_); }
FWD_SHAPE(MatrixMultiply) { *y[0] = shape_ops::matmul(*x[0], *x[1]); }
FWD_SHAPE(Max) { *y[0] = x[0]->resize_dim(dim_, 1); }
FWD_SHAPE(Min) { *y[0] = x[0]->resize_dim(dim_, 1); }
FWD_SHAPE(Sum) { *y[0] = x[0]->resize_dim(dim_, 1); }
FWD_SHAPE(LogSumExp) { *y[0] = x[0]->resize_dim(dim_, 1); }
FWD_SHAPE(Broadcast) { *y[0] = shape_ops::broadcast(*x[0], dim_, size_); }
FWD_SHAPE(BatchPick) { *y[0] = shape_ops::batch_pick(*x[0], ids_); }
FWD_SHAPE(BatchSlice) { *y[0] = shape_ops::batch_slice(*x[0], lower_, upper_); }
FWD_SHAPE(BatchSplit) {
  if (n_ == 0) {
    PRIMITIV_THROW_ERROR("Invalid number of partitions: " << n_);
  }
  Shape xs = *x[0];
  const std::uint32_t total = xs.batch();
  const std::uint32_t span = total / n_;
  if (span * n_ != total) {
    PRIMITIV_THROW_ERROR(
        "Could not split the batch with size "
        << total << " into " << n_ << " partitions.");
  }
  xs.update_batch(span);
  for (std::uint32_t i = 0; i < n_; ++i) {
    *y[i] = xs;
  }
}
FWD_SHAPE(BatchConcat) { *y[0] = shape_ops::batch_concat(x); }
FWD_SHAPE(BatchSum) { *y[0] = x[0]->resize_batch(1); }
FWD_SHAPE(Convolution2D) {
  *y[0] = shape_ops::conv2d(
      *x[0], *x[1],
      padding0_, padding1_, stride0_, stride1_, dilation0_, dilation1_);
}
FWD_SHAPE(MaxPooling2D) {
  *y[0] = shape_ops::pool2d(
      *x[0], window0_, window1_, padding0_, padding1_, stride0_, stride1_);
}
FWD_SHAPE(SoftmaxCrossEntropy) {
  *y[0] = shape_ops::elementwise(*x[0], *x[1]);
  y[0]->update_dim(dim_, 1);
}
FWD_SHAPE(SparseSoftmaxCrossEntropy) {
  *y[0] = shape_ops::pick(*x[0], ids_, dim_);
}
FWD_SHAPE_UNARY(StopGradient);

#undef FWD_SHAPE_UNARY
#undef FWD_SHAPE_SCALAR
#undef FWD_SHAPE_ELEMENTWISE
#undef FWD_SHAPE

/*
 * Obtaining inner values.
 */

vector<const Tensor *> Parameter::get_inner_values() const {
  return std::vector<const Tensor *> { &param_.value() };
}

/*
 * Forward operations.
 */

#define FORWARD(name) \
  void name::forward( \
      const vector<const Tensor *> &x, \
      const vector<Tensor *> &y) const

FORWARD(Input) {
  UNUSED(x);
  *y[0] = functions::input<Tensor>(shape_, data_, device_);
}

FORWARD(Copy) { *y[0] = functions::copy(*x[0], device_); }

FORWARD(Constant) {
  UNUSED(x);
  *y[0] = functions::constant<Tensor>(shape_, k_, device_);
}

FORWARD(Identity) {
  UNUSED(x);
  *y[0] = device_.identity(size_);
}

FORWARD(RandomBernoulli) {
  UNUSED(x);
  *y[0] = device_.random_bernoulli(shape_, p_);
}
FORWARD(RandomUniform) {
  UNUSED(x);
  *y[0] = device_.random_uniform(shape_, lower_, upper_);
}
FORWARD(RandomNormal) {
  UNUSED(x);
  *y[0] = device_.random_normal(shape_, mean_, sd_);
}
FORWARD(RandomLogNormal) {
  UNUSED(x);
  *y[0] = device_.random_log_normal(shape_, mu_, beta_);
}

FORWARD(Pick) { *y[0] = functions::pick(*x[0], ids_, dim_); }
FORWARD(Slice) { *y[0] = functions::slice(*x[0], dim_, lower_, upper_); }
FORWARD(Split) {
  const std::uint32_t total = x[0]->shape()[dim_];
  const std::uint32_t span = total / n_;
  for (std::uint32_t i = 0; i < n_; ++i) {
    *y[i] = functions::slice(*x[0], dim_, i * span, (i + 1) * span);
  }
}
FORWARD(Concat) { *y[0] = functions::concat(x, dim_); }

FORWARD(Reshape) { *y[0] = functions::reshape(*x[0], shape_); }
FORWARD(Flatten) { *y[0] = functions::flatten(*x[0]); }

FORWARD(Positive) { *y[0] = *x[0]; }
FORWARD(Negative) { *y[0] = -(*x[0]); }
FORWARD(Abs) { *y[0] = functions::abs(*x[0]); }
FORWARD(Sqrt) { *y[0] = functions::sqrt(*x[0]); }
FORWARD(Exp) { *y[0] = functions::exp(*x[0]); }
FORWARD(Log) { *y[0] = functions::log(*x[0]); }
FORWARD(Tanh) { *y[0] = functions::tanh(*x[0]); }
FORWARD(Sigmoid) { *y[0] = functions::sigmoid(*x[0]); }
FORWARD(Softplus) { *y[0] = functions::softplus(*x[0]); }
FORWARD(Sin) { *y[0] = functions::sin(*x[0]); }
FORWARD(Cos) { *y[0] = functions::cos(*x[0]); }
FORWARD(Tan) { *y[0] = functions::tan(*x[0]); }
FORWARD(ReLU) { *y[0] = functions::relu(*x[0]); }
FORWARD(LReLU) { *y[0] = functions::lrelu(*x[0]); }

FORWARD(AddConst) { *y[0] = *x[0] + k_; }
FORWARD(SubtractConstR) { *y[0] = *x[0] - k_; }
FORWARD(SubtractConstL) { *y[0] = k_ - *x[0]; }
FORWARD(MultiplyConst) { *y[0] = *x[0] * k_; }
FORWARD(DivideConstR) { *y[0] = *x[0] / k_; }
FORWARD(DivideConstL) { *y[0] = k_ / *x[0]; }
FORWARD(PowConstR) { *y[0] = functions::pow(*x[0], k_); }
FORWARD(PowConstL) { *y[0] = functions::pow(k_, *x[0]); }
FORWARD(PReLU) { *y[0] = functions::prelu(*x[0], k_); }
FORWARD(ELU) { *y[0] = functions::elu(*x[0], k_); }

FORWARD(PowN) { *y[0] = functions::pown(*x[0], k_); }

FORWARD(AddScalar) { *y[0] = *x[0] + *x[1]; }
FORWARD(SubtractScalarR) { *y[0] = *x[0] - *x[1]; }
FORWARD(SubtractScalarL) { *y[0] = *x[1] - *x[0]; }
FORWARD(MultiplyScalar) { *y[0] = *x[0] * *x[1]; }
FORWARD(DivideScalarR) { *y[0] = *x[0] / *x[1]; }
FORWARD(DivideScalarL) { *y[0] = *x[1] / *x[0]; }
FORWARD(PowScalarR) { *y[0] = functions::pow(*x[0], *x[1]); }
FORWARD(PowScalarL) { *y[0] = functions::pow(*x[1], *x[0]); }

FORWARD(Add) { *y[0] = *x[0] + *x[1]; }
FORWARD(Subtract) { *y[0] = *x[0] - *x[1]; }
FORWARD(Multiply) { *y[0] = *x[0] * *x[1]; }
FORWARD(Divide) { *y[0] = *x[0] / *x[1]; }
FORWARD(Pow) { *y[0] = functions::pow(*x[0], *x[1]); }

FORWARD(Transpose) { *y[0] = functions::transpose(*x[0]); }
FORWARD(PermuteDims) { *y[0] = functions::permute_dims(*x[0], perm_); }
FORWARD(MatrixMultiply) { *y[0] = functions::matmul(*x[0], *x[1]); }

FORWARD(Flip) { *y[0] = functions::flip(*x[0], dim_); }

FORWARD(Sum) { *y[0] = functions::sum(*x[0], dim_); }
FORWARD(LogSumExp) { *y[0] = functions::logsumexp(*x[0], dim_); }
FORWARD(Broadcast) { *y[0] = functions::broadcast(*x[0], dim_, size_); }
  
FORWARD(Max) { *y[0] = functions::max(*x[0], dim_); }
FORWARD(Min) { *y[0] = functions::min(*x[0], dim_); }

FORWARD(BatchPick) { *y[0] = functions::batch::pick(*x[0], ids_); }
FORWARD(BatchSlice) { *y[0] = functions::batch::slice(*x[0], lower_, upper_); }
FORWARD(BatchSplit) {
  const std::uint32_t total = x[0]->shape().batch();
  const std::uint32_t span = total / n_;
  for (std::uint32_t i = 0; i < n_; ++i) {
    *y[i] = functions::batch::slice(*x[0], i * span, (i + 1) * span);
  }
}
FORWARD(BatchConcat) { *y[0] = functions::batch::concat(x); }

FORWARD(BatchSum) { *y[0] = functions::batch::sum(*x[0]); }

FORWARD(Convolution2D) {
  *y[0] = functions::conv2d(
      *x[0], *x[1],
      padding0_, padding1_, stride0_, stride1_, dilation0_, dilation1_);
}

FORWARD(MaxPooling2D) {
  *y[0] = functions::max_pool2d(
      *x[0], window0_, window1_, padding0_, padding1_, stride0_, stride1_);
}

FORWARD(SoftmaxCrossEntropy) {
  *y[0] = functions::softmax_cross_entropy(*x[0], *x[1], dim_);
}
FORWARD(SparseSoftmaxCrossEntropy) {
#ifdef PRIMITIV_USE_CACHE
  log_softmax_x_ = functions::log_softmax(*x[0], dim_);
  *y[0] = functions::pick(-log_softmax_x_, ids_, dim_);
#else
  *y[0] = functions::softmax_cross_entropy(*x[0], ids_, dim_);
#endif  // PRIMITIV_USE_CACHE
}

FORWARD(StopGradient) { *y[0] = *x[0]; }

#undef FORWARD

/*
 * Backward operations.
 */

#define BACKWARD(name) \
  void name::backward( \
      const vector<const Tensor *> &x, \
      const vector<const Tensor *> &y, \
      const vector<const Tensor *> &gy, \
      const vector<Tensor *> &gx) const

#define BACKWARD_NOP(name) \
  BACKWARD(name) { UNUSED(x); UNUSED(y); UNUSED(gy); UNUSED(gx); }

BACKWARD_NOP(Input);

BACKWARD(Parameter) {
  UNUSED(x);
  UNUSED(y);
  UNUSED(gx);
  param_.gradient() += *gy[0];
}

BACKWARD(Copy) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] += functions::copy(*gy[0], gx[0]->device());
}

BACKWARD_NOP(Constant);
BACKWARD_NOP(Identity);
BACKWARD_NOP(RandomBernoulli);
BACKWARD_NOP(RandomUniform);
BACKWARD_NOP(RandomNormal);
BACKWARD_NOP(RandomLogNormal);

BACKWARD(Pick) {
  UNUSED(x);
  UNUSED(y);
  gy[0]->device().pick_bw(*gy[0], ids_, dim_, *gx[0]);
}

BACKWARD(Slice) {
  UNUSED(x);
  UNUSED(y);
  gy[0]->device().slice_bw(*gy[0], dim_, lower_, *gx[0]);
}

BACKWARD(Split) {
  UNUSED(x);
  UNUSED(y);
  Device &dev = gy[0]->device();
  const std::uint32_t span = gy[0]->shape()[dim_];
  for (std::uint32_t i = 0; i < n_; ++i) {
    dev.slice_bw(*gy[i], dim_, i * span, *gx[0]);
  }
}

BACKWARD(Concat) {
  UNUSED(x);
  UNUSED(y);
  std::uint32_t offset = 0;
  for (Tensor *gxi : gx) {
    const std::uint32_t span = gxi->shape()[dim_];
    *gxi += functions::slice(*gy[0], dim_, offset, offset + span);
    offset += span;
  }
}

BACKWARD(Reshape) {
  UNUSED(y);
  *gx[0] += gy[0]->reshape(x[0]->shape());
}

BACKWARD(Flatten) {
  UNUSED(y);
  *gx[0] += gy[0]->reshape(x[0]->shape());
}

BACKWARD(Positive) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] += *gy[0];
}

BACKWARD(Negative) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] -= *gy[0];
}

BACKWARD(Abs) {
  gy[0]->device().abs_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Sqrt) {
  gy[0]->device().sqrt_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Exp) {
  gy[0]->device().exp_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Log) {
  gy[0]->device().log_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Tanh) {
  gy[0]->device().tanh_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Sigmoid) {
  gy[0]->device().sigmoid_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Softplus) {
  gy[0]->device().softplus_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Sin) {
  gy[0]->device().sin_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Cos) {
  gy[0]->device().cos_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(Tan) {
  gy[0]->device().tan_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(ReLU) {
  gy[0]->device().prelu_bw(*x[0], *y[0], *gy[0], 0, *gx[0]);
}

BACKWARD(LReLU) {
  gy[0]->device().prelu_bw(*x[0], *y[0], *gy[0], .01, *gx[0]);
}

BACKWARD(Transpose) {
  gy[0]->device().transpose_bw(*x[0], *y[0], *gy[0], *gx[0]);
}

BACKWARD(PermuteDims) {
  gy[0]->device().permute_dims_bw(*x[0], *y[0], *gy[0], perm_, *gx[0]);
}

BACKWARD(AddConst) {
  gy[0]->device().add_const_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(SubtractConstR) {
  gy[0]->device().subtract_const_r_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(SubtractConstL) {
  gy[0]->device().subtract_const_l_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(MultiplyConst) {
  gy[0]->device().multiply_const_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(DivideConstR) {
  gy[0]->device().divide_const_r_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(DivideConstL) {
  gy[0]->device().divide_const_l_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(PowConstR) {
  gy[0]->device().pow_const_r_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(PowConstL) {
  gy[0]->device().pow_const_l_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(PReLU) {
  gy[0]->device().prelu_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(ELU) {
  gy[0]->device().elu_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(PowN) {
  gy[0]->device().pown_bw(*x[0], *y[0], *gy[0], k_, *gx[0]);
}

BACKWARD(AddScalar) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] += *gy[0];
  *gx[1] += functions::sum(gy[0]->flatten(), 0);
}

BACKWARD(SubtractScalarR) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] += *gy[0];
  *gx[1] -= functions::sum(gy[0]->flatten(), 0);
}

BACKWARD(SubtractScalarL) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] -= *gy[0];
  *gx[1] += functions::sum(gy[0]->flatten(), 0);
}

BACKWARD(MultiplyScalar) {
  UNUSED(y);
  *gx[0] += *x[1] * *gy[0];
  *gx[1] += functions::sum((*x[0] * *gy[0]).flatten(), 0);
}

BACKWARD(DivideScalarR) {
  const Tensor a = *gy[0] / *x[1];
  *gx[0] += a;
  *gx[1] -= functions::sum((a * *y[0]).flatten(), 0);
}

BACKWARD(DivideScalarL) {
  const Tensor a = *gy[0] / *x[0];
  *gx[0] -= a * *y[0];
  *gx[1] += functions::sum(a.flatten(), 0);
}

BACKWARD(PowScalarR) {
  const Tensor a = *gy[0] * *y[0];
  *gx[0] += a * *x[1] / *x[0];
  *gx[1] += functions::sum((a * functions::log(*x[0])).flatten(), 0);
}

BACKWARD(PowScalarL) {
  const Tensor a = *gy[0] * *y[0];
  *gx[0] += a * functions::log(*x[1]);
  *gx[1] += functions::sum((a * *x[0] / *x[1]).flatten(), 0);
}

BACKWARD(Add) {
  gy[0]->device().add_bw(*x[0], *x[1], *y[0], *gy[0], *gx[0], *gx[1]);
}

BACKWARD(Subtract) {
  gy[0]->device().subtract_bw(*x[0], *x[1], *y[0], *gy[0], *gx[0], *gx[1]);
}

BACKWARD(Multiply) {
  gy[0]->device().multiply_bw(*x[0], *x[1], *y[0], *gy[0], *gx[0], *gx[1]);
}

BACKWARD(Divide) {
  gy[0]->device().divide_bw(*x[0], *x[1], *y[0], *gy[0], *gx[0], *gx[1]);
}

BACKWARD(Pow) {
  gy[0]->device().pow_bw(*x[0], *x[1], *y[0], *gy[0], *gx[0], *gx[1]);
}

BACKWARD(MatrixMultiply) {
  gy[0]->device().matmul_bw(*x[0], *x[1], *y[0], *gy[0], *gx[0], *gx[1]);
}

BACKWARD(Flip) {
  UNUSED(x);
  UNUSED(y);
  gy[0]->device().flip_bw(*gy[0], dim_, *gx[0]);
}

BACKWARD(Max) {
  gy[0]->device().max_bw(*x[0], *y[0], *gy[0], dim_, *gx[0]);
}

BACKWARD(Min) {
  gy[0]->device().min_bw(*x[0], *y[0], *gy[0], dim_, *gx[0]);
}

BACKWARD(Sum) {
  UNUSED(y);
  *gx[0] += functions::broadcast(*gy[0], dim_, x[0]->shape()[dim_]);
}

BACKWARD(LogSumExp) {
  // NOTE(odashi): dy/dx = softmax(x) = exp(x - y)
  const std::uint32_t n = x[0]->shape()[dim_];
  *gx[0]
    += functions::exp(*x[0] - functions::broadcast(*y[0], dim_, n))
    * functions::broadcast(*gy[0], dim_, n);
}

BACKWARD(Broadcast) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] += functions::sum(*gy[0], dim_);
}

BACKWARD(BatchPick) {
  UNUSED(x);
  UNUSED(y);
  gy[0]->device().batch_pick_bw(*gy[0], ids_, *gx[0]);
}

BACKWARD(BatchSlice) {
  UNUSED(x);
  UNUSED(y);
  gy[0]->device().batch_slice_bw(*gy[0], lower_, *gx[0]);
}

BACKWARD(BatchSplit) {
  UNUSED(x);
  UNUSED(y);
  Device &dev = gy[0]->device();
  const std::uint32_t span = gy[0]->shape().batch();
  for (std::uint32_t i = 0; i < n_; ++i) {
    dev.batch_slice_bw(*gy[i], i * span, *gx[0]);
  }
}

BACKWARD(BatchConcat) {
  UNUSED(x);
  UNUSED(y);
  std::uint32_t offset = 0;
  for (Tensor *gxi : gx) {
    const std::uint32_t span = gxi->shape().batch();
    *gxi += functions::batch::slice(*gy[0], offset, offset + span);
    offset += span;
  }
}

BACKWARD(BatchSum) {
  UNUSED(x);
  UNUSED(y);
  *gx[0] += *gy[0];
}

BACKWARD(Convolution2D) {
  gy[0]->device().conv2d_bw(
      *x[0], *x[1], *y[0], *gy[0],
      padding0_, padding1_, stride0_, stride1_, dilation0_, dilation1_,
      *gx[0], *gx[1]);
}

BACKWARD(MaxPooling2D) {
  gy[0]->device().max_pool2d_bw(
      *x[0], *y[0], *gy[0],
      window0_, window1_, padding0_, padding1_, stride0_, stride1_,
      *gx[0]);
}

BACKWARD(SoftmaxCrossEntropy) {
  UNUSED(y);
  const Tensor log_softmax_x = functions::log_softmax(*x[0], dim_);
  const Tensor bcast_gy = functions::broadcast(
      *gy[0], dim_, x[0]->shape()[dim_]);
  *gx[0] += (functions::exp(log_softmax_x) - *x[1]) * bcast_gy;
  *gx[1] -= log_softmax_x * bcast_gy;
}

BACKWARD(SparseSoftmaxCrossEntropy) {
  // dE/dx = gy * (softmax(x) - delta(x, i))
  //       = gy * softmax(x) - gy * delta(x, i)
  UNUSED(y);
#ifdef PRIMITIV_USE_CACHE
  *gx[0]
    += functions::exp(log_softmax_x_)
    * functions::broadcast(*gy[0], dim_, x[0]->shape()[dim_]);
#else
  *gx[0]
    += functions::softmax(*x[0], dim_)
    * functions::broadcast(*gy[0], dim_, x[0]->shape()[dim_]);
#endif  // PRIMITIV_USE_CACHE
  gy[0]->device().pick_bw(-*gy[0], ids_, dim_, *gx[0]);
}

BACKWARD_NOP(StopGradient);

#undef BACKWARD_NOP
#undef BACKWARD

}  // namespace operators
}  // namespace primitive

#undef UNUSED
