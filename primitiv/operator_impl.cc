#include <primitiv/config.h>

#include <algorithm>
#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/operator_impl.h>
#include <primitiv/parameter.h>
#include <primitiv/shape_ops.h>

using std::vector;

namespace primitiv {
namespace operators {

#define CHECK_ARGNUM(args, n) \
  if (args.size() != n) { \
    THROW_ERROR( \
        "Number of arguments mismatched." \
        << " operator: " << name() \
        << ", required: " << n \
        << " != actual: " << args.size()); \
  }

Input::Input(const Shape &shape, const vector<float> &data, Device &device)
: shape_(shape)
, data_(data)
, device_(device) {
  if (data_.size() != shape_.size()) {
    THROW_ERROR(
        "Data sizes mismatched."
        << " operator: Input"
        << ", required: " << shape_.size() << " (" << shape_.to_string() << ")"
        << ", actual: " << data_.size());
  }
}

Shape Input::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor Input::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return functions::input<Tensor>(shape_, data_, device_);
}

void Input::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape ParameterInput::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return param_.shape();
}

Tensor ParameterInput::forward(const vector<const Tensor *> &args) {
  THROW_ERROR(
      "Attempted to get return values of ParameterInput via forward().");
}

void ParameterInput::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  param_.gradient() += cur_grad;
}

Shape Copy::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return *args[0];
}

Tensor Copy::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 1);
  return functions::copy(*args[0], device_);
}

void Copy::backward(
    const Tensor &y, const Tensor &gy,
    const vector<const Tensor *> &x, const vector<Tensor *> &gx) const {
  *gx[0] += functions::copy(gy, gx[0]->device());
}

Shape Constant::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor Constant::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return functions::constant<Tensor>(shape_, k_, device_);
}

void Constant::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape IdentityMatrix::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return Shape({size_, size_});
}

Tensor IdentityMatrix::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return device_.identity(size_);
}

void IdentityMatrix::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape RandomBernoulli::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor RandomBernoulli::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return device_.random_bernoulli(shape_, p_);
}

void RandomBernoulli::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape RandomUniform::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor RandomUniform::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return device_.random_uniform(shape_, lower_, upper_);
}

void RandomUniform::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape RandomNormal::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor RandomNormal::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return device_.random_normal(shape_, mean_, sd_);
}

void RandomNormal::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape RandomLogNormal::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor RandomLogNormal::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 0);
  return device_.random_log_normal(shape_, mean_, sd_);
}

void RandomLogNormal::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape Pick::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::pick(*args[0], ids_, dim_);
}

Tensor Pick::forward(const vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 1);
  return functions::pick(*args[0], ids_, dim_);
}

void Pick::backward(
    const Tensor &y, const Tensor &gy,
    const vector<const Tensor *> &x, const vector<Tensor *> &gx) const {
  gy.device().pick_bw(gy, ids_, dim_, *gx[0]);
}

Shape Slice::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::slice(*args[0], dim_, lower_, upper_);
}

Tensor Slice::forward(const std::vector<const Tensor *> &args) {
  CHECK_ARGNUM(args, 1);
  return functions::slice(*args[0], dim_, lower_, upper_);
}

void Slice::backward(
    const Tensor &y, const Tensor &gy,
    const vector<const Tensor *> &x, const vector<Tensor *> &gx) const {
  gy.device().slice_bw(gy, dim_, lower_, *gx[0]);
}

Shape Concat::forward_shape(const vector<const Shape *> &args) const {
  return shape_ops::concat(args, dim_);
}

Tensor Concat::forward(const std::vector<const Tensor *> &args) {
  return functions::concat(args, dim_);
}

void Concat::backward(
    const Tensor &y, const Tensor &gy,
    const vector<const Tensor *> &x, const vector<Tensor *> &gx) const {
  std::uint32_t offset = 0;
  for (Tensor *gxi : gx) {
    const std::uint32_t span = gxi->shape()[dim_];
    *gxi += functions::slice(gy, dim_, offset, offset + span);
    offset += span;
  }
}

Shape Reshape::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::reshape(*args[0], shape_);
}

Shape Flatten::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::flatten(*args[0]);
}

#define FWD_SHAPE_UNARY(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 1); \
    return *args[0]; \
  }

#define FWD_SHAPE_SCALAR(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 2); \
    return shape_ops::scalar_op(*args[0], *args[1]); \
  }

#define FWD_SHAPE_ELEMENTWISE(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 2); \
    return shape_ops::elementwise(*args[0], *args[1]); \
  }

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

#undef FWD_SHAPE_UNARY
#undef FWD_SHAPE_ELEMENTWISE

Shape Transpose::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::transpose(*args[0]);
}

Shape MatrixMultiply::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 2);
  return shape_ops::matmul(*args[0], *args[1]);
}

Shape Sum::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return args[0]->resize_dim(dim_, 1);
}

Shape LogSumExp::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return args[0]->resize_dim(dim_, 1);
}

Shape Broadcast::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::broadcast(*args[0], dim_, size_);
}

Shape BatchSum::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return args[0]->resize_batch(1);
}

Shape SoftmaxCrossEntropy::forward_shape(
    const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 2);
  Shape y = shape_ops::elementwise(*args[0], *args[1]);
  y.update_dim(dim_, 1);
  return y;
}

Shape SparseSoftmaxCrossEntropy::forward_shape(
    const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::pick(*args[0], ids_, dim_);
}

Shape StopGradient::forward_shape(
    const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return *args[0];
}

#define FORWARD(name) \
    Tensor name::forward(const vector<const Tensor *> &x)

FORWARD(Reshape) { return functions::reshape(*x[0], shape_); }
FORWARD(Flatten) { return functions::flatten(*x[0]); }

FORWARD(Positive) { return *x[0]; }
FORWARD(Negative) { return -(*x[0]); }
FORWARD(Sqrt) { return functions::sqrt(*x[0]); }
FORWARD(Exp) { return functions::exp(*x[0]); }
FORWARD(Log) { return functions::log(*x[0]); }
FORWARD(Tanh) { return functions::tanh(*x[0]); }
FORWARD(Sigmoid) { return functions::sigmoid(*x[0]); }
FORWARD(Softplus) { return functions::softplus(*x[0]); }
FORWARD(Sin) { return functions::sin(*x[0]); }
FORWARD(Cos) { return functions::cos(*x[0]); }
FORWARD(Tan) { return functions::tan(*x[0]); }
FORWARD(ReLU) { return functions::relu(*x[0]); }
FORWARD(LReLU) { return functions::lrelu(*x[0]); }

FORWARD(AddConst) { return *x[0] + k_; }
FORWARD(SubtractConstR) { return *x[0] - k_; }
FORWARD(SubtractConstL) { return k_ - *x[0]; }
FORWARD(MultiplyConst) { return *x[0] * k_; }
FORWARD(DivideConstR) { return *x[0] / k_; }
FORWARD(DivideConstL) { return k_ / *x[0]; }
FORWARD(PowConstR) { return functions::pow(*x[0], k_); }
FORWARD(PowConstL) { return functions::pow(k_, *x[0]); }
FORWARD(PReLU) { return functions::prelu(*x[0], k_); }
FORWARD(ELU) { return functions::elu(*x[0], k_); }

FORWARD(AddScalar) { return *x[0] + *x[1]; }
FORWARD(SubtractScalarR) { return *x[0] - *x[1]; }
FORWARD(SubtractScalarL) { return *x[1] - *x[0]; }
FORWARD(MultiplyScalar) { return *x[0] * *x[1]; }
FORWARD(DivideScalarR) { return *x[0] / *x[1]; }
FORWARD(DivideScalarL) { return *x[1] / *x[0]; }
FORWARD(PowScalarR) { return functions::pow(*x[0], *x[1]); }
FORWARD(PowScalarL) { return functions::pow(*x[1], *x[0]); }

FORWARD(Add) { return *x[0] + *x[1]; }
FORWARD(Subtract) { return *x[0] - *x[1]; }
FORWARD(Multiply) { return *x[0] * *x[1]; }
FORWARD(Divide) { return *x[0] / *x[1]; }
FORWARD(Pow) { return functions::pow(*x[0], *x[1]); }

FORWARD(Transpose) { return functions::transpose(*x[0]); }
FORWARD(MatrixMultiply) { return functions::matmul(*x[0], *x[1]); }

FORWARD(Sum) { return functions::sum(*x[0], dim_); }
FORWARD(LogSumExp) { return functions::logsumexp(*x[0], dim_); }
FORWARD(Broadcast) { return functions::broadcast(*x[0], dim_, size_); }

FORWARD(BatchSum) { return functions::batch::sum(*x[0]); }

FORWARD(SoftmaxCrossEntropy) {
  return functions::softmax_cross_entropy(*x[0], *x[1], dim_);
}
FORWARD(SparseSoftmaxCrossEntropy) {
#ifdef PRIMITIV_USE_CACHE
  log_softmax_x_ = functions::log_softmax(*x[0], dim_);
  return functions::pick(-log_softmax_x_, ids_, dim_);
#else
  return functions::softmax_cross_entropy(*x[0], ids_, dim_);
#endif  // PRIMITIV_USE_CACHE
}

FORWARD(StopGradient) { return *x[0]; }

#undef FORWARD

#define BACKWARD(name) \
  void name::backward( \
      const Tensor &y, \
      const Tensor &gy, \
      const vector<const Tensor *> &x, \
      const vector<Tensor *> &gx) const


BACKWARD(Reshape) { *gx[0] += gy.reshape(x[0]->shape()); }
BACKWARD(Flatten) { *gx[0] += gy.reshape(x[0]->shape()); }

BACKWARD(Positive) { *gx[0] += gy; }
BACKWARD(Negative) { *gx[0] -= gy; }
BACKWARD(Sqrt) { gy.device().sqrt_bw(*x[0], y, gy, *gx[0]); }
BACKWARD(Exp) {  gy.device().exp_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Log) {  gy.device().log_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Tanh) {  gy.device().tanh_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Sigmoid) {  gy.device().sigmoid_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Softplus) {  gy.device().softplus_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Sin) {  gy.device().sin_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Cos) {  gy.device().cos_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(Tan) {  gy.device().tan_bw(*x[0], y, gy, *gx[0]);}
BACKWARD(ReLU) { gy.device().prelu_bw(*x[0], y, gy, 0, *gx[0]); }
BACKWARD(LReLU) { gy.device().prelu_bw(*x[0], y, gy, .01, *gx[0]); }
BACKWARD(Transpose) { gy.device().transpose_bw(*x[0], y, gy, *gx[0]); }

BACKWARD(AddConst) { gy.device().add_const_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(SubtractConstR) { gy.device().subtract_const_r_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(SubtractConstL) { gy.device().subtract_const_l_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(MultiplyConst) { gy.device().multiply_const_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(DivideConstR) { gy.device().divide_const_r_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(DivideConstL) { gy.device().divide_const_l_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(PowConstR) { gy.device().pow_const_r_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(PowConstL) { gy.device().pow_const_l_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(PReLU) { gy.device().prelu_bw(*x[0], y, gy, k_, *gx[0]); }
BACKWARD(ELU) { gy.device().elu_bw(*x[0], y, gy, k_, *gx[0]); }

BACKWARD(AddScalar) {
  *gx[0] += gy;
  *gx[1] += functions::sum(gy.flatten(), 0);
}
BACKWARD(SubtractScalarR) {
  *gx[0] += gy;
  *gx[1] -= functions::sum(gy.flatten(), 0);
}
BACKWARD(SubtractScalarL) {
  *gx[0] -= gy;
  *gx[1] += functions::sum(gy.flatten(), 0);
}
BACKWARD(MultiplyScalar) {
  *gx[0] += *x[1] * gy;
  *gx[1] += functions::sum((*x[0] * gy).flatten(), 0);
}
BACKWARD(DivideScalarR) {
  const Tensor a = gy / *x[1];
  *gx[0] += a;
  *gx[1] -= functions::sum((a * y).flatten(), 0);
}
BACKWARD(DivideScalarL) {
  const Tensor a = gy / *x[0];
  *gx[0] -= a * y;
  *gx[1] += functions::sum(a.flatten(), 0);
}
BACKWARD(PowScalarR) {
  const Tensor a = gy * y;
  *gx[0] += a * *x[1] / *x[0];
  *gx[1] += functions::sum((a * functions::log(*x[0])).flatten(), 0);
}
BACKWARD(PowScalarL) {
  const Tensor a = gy * y;
  *gx[0] += a * functions::log(*x[1]);
  *gx[1] += functions::sum((a * *x[0] / *x[1]).flatten(), 0);
}

BACKWARD(Add) { gy.device().add_bw(*x[0], *x[1], y, gy, *gx[0], *gx[1]); }
BACKWARD(Subtract) { gy.device().subtract_bw(*x[0], *x[1], y, gy, *gx[0], *gx[1]); }
BACKWARD(Multiply) { gy.device().multiply_bw(*x[0], *x[1], y, gy, *gx[0], *gx[1]); }
BACKWARD(Divide) { gy.device().divide_bw(*x[0], *x[1], y, gy, *gx[0], *gx[1]); }
BACKWARD(Pow) { gy.device().pow_bw(*x[0], *x[1], y, gy, *gx[0], *gx[1]); }
BACKWARD(MatrixMultiply) { gy.device().matmul_bw(*x[0], *x[1], y, gy, *gx[0], *gx[1]); }

BACKWARD(Sum) { *gx[0] += functions::broadcast(gy, dim_, x[0]->shape()[dim_]); }
BACKWARD(LogSumExp) {
  // NOTE(odashi): dy/dx = softmax(x) = exp(x - y)
  const std::uint32_t n = x[0]->shape()[dim_];
  *gx[0]
    += functions::exp(*x[0] - functions::broadcast(y, dim_, n))
    * functions::broadcast(gy, dim_, n);
}
BACKWARD(Broadcast) { *gx[0] += functions::sum(gy, dim_); }

BACKWARD(BatchSum) { *gx[0] += gy; }

BACKWARD(SoftmaxCrossEntropy) {
  const Tensor log_softmax_x = functions::log_softmax(*x[0], dim_);
  const Tensor bcast_gy = functions::broadcast(gy, dim_, x[0]->shape()[dim_]);
  *gx[0] += (functions::exp(log_softmax_x) - *x[1]) * bcast_gy;
  *gx[1] -= log_softmax_x * bcast_gy;
}

BACKWARD(SparseSoftmaxCrossEntropy) {
  // dE/dx = gy * (softmax(x) - delta(x, i))
  //       = gy * softmax(x) - gy * delta(x, i)
#ifdef PRIMITIV_USE_CACHE
  *gx[0]
    += functions::exp(log_softmax_x_)
    * functions::broadcast(gy, dim_, x[0]->shape()[dim_]);
#else
  *gx[0]
    += functions::softmax(*x[0], dim_)
    * functions::broadcast(gy, dim_, x[0]->shape()[dim_]);
#endif  // PRIMITIV_USE_CACHE
  gy.device().pick_bw(-gy, ids_, dim_, *gx[0]);
}

BACKWARD(StopGradient) {}

#undef BACKWARD

}  // namespace operators
}  // namespace primitive
