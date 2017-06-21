#include <config.h>

#include <algorithm>
#include <primitiv/error.h>
#include <primitiv/function_impl.h>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>
#include <primitiv/shape_ops.h>

using std::vector;
namespace T = primitiv::tensor_ops;

namespace primitiv {
namespace functions {

#define CHECK_ARGNUM(args, n) \
  if (args.size() != n) { \
    THROW_ERROR( \
        "Number of arguments mismatched." \
        << " function: " << name() \
        << ", required: " << n \
        << " != actual: " << args.size()); \
  }

Input::Input(const Shape &shape, Device *device, const vector<float> &data)
: shape_(shape)
, device_(device)
, data_(data) {
  const unsigned shape_size = shape_.num_total_elements();
  if (data_.size() != shape_size) {
    THROW_ERROR(
        "Data sizes mismatched."
        << " function: Input"
        << ", required: " << shape_size << " (" << shape_.to_string() << ")"
        << ", actual: " << data_.size());
  }
}

Shape Input::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor Input::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  Tensor ret = device_->new_tensor(shape_);
  ret.reset(data_);
  return ret;
}

void Input::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do
}

Shape ParameterInput::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return param_->shape();
}

Tensor ParameterInput::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return +param_->value();
}

void ParameterInput::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  param_->add_gradient(cur_grad);
}

Shape Pick::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::pick(*args[0], dim_, ids_);
}

Tensor Pick::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 1);
  return tensor_ops::pick(*args[0], dim_, ids_);
}

void Pick::backward(
    const Tensor &y, const Tensor &yg,
    const vector<const Tensor *> &x, const vector<Tensor *> &xg) const {
  xg[0]->add_gradient_sparse(yg, dim_, ids_);
}

Shape Slice::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::slice(*args[0], dim_, lower_, upper_);
}

Tensor Slice::forward(const std::vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 1);
  return T::slice(*args[0], dim_, lower_, upper_);
}

void Slice::backward(
    const Tensor &y, const Tensor &yg,
    const vector<const Tensor *> &x, const vector<Tensor *> &xg) const {
  xg[0]->add_gradient_offset(yg, dim_, lower_);
}

Shape Concat::forward_shape(const vector<const Shape *> &args) const {
  return shape_ops::concat(args, dim_);
}

Tensor Concat::forward(const std::vector<const Tensor *> &args) const {
  return T::concat(args, dim_);
}

void Concat::backward(
    const Tensor &y, const Tensor &yg,
    const vector<const Tensor *> &x, const vector<Tensor *> &xg) const {
  unsigned offset = 0;
  for (Tensor *xgi : xg) {
    const unsigned span = xgi->shape()[dim_];
    xgi->add_gradient(T::slice(yg, dim_, offset, offset + span));
    offset += span;
  }
}

#define FWD_SHAPE_UNARY(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 1); \
    return *args[0]; \
  }

#define FWD_SHAPE_ARITHMETIC(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 2); \
    return shape_ops::elementwise(*args[0], *args[1]); \
  }

FWD_SHAPE_UNARY(Positive);
FWD_SHAPE_UNARY(Negative);
FWD_SHAPE_UNARY(AddConst);
FWD_SHAPE_UNARY(SubtractConstL);
FWD_SHAPE_UNARY(SubtractConstR);
FWD_SHAPE_UNARY(MultiplyConst);
FWD_SHAPE_UNARY(DivideConstL);
FWD_SHAPE_UNARY(DivideConstR);
FWD_SHAPE_UNARY(Exp);
FWD_SHAPE_UNARY(Tanh);
FWD_SHAPE_UNARY(Sigmoid);
FWD_SHAPE_UNARY(ReLU);
FWD_SHAPE_ARITHMETIC(Add);
FWD_SHAPE_ARITHMETIC(Subtract);
FWD_SHAPE_ARITHMETIC(Multiply);
FWD_SHAPE_ARITHMETIC(Divide);

Shape Transpose::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::transpose(*args[0]);
}

Shape Dot::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 2);
  return shape_ops::dot(*args[0], *args[1]);
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

#undef FWD_SHAPE_UNARY
#undef FWD_SHAPE_ARITHMETIC

#define FORWARD(name) \
    Tensor name::forward(const vector<const Tensor *> &x) const

FORWARD(Positive) { return +(*x[0]); }
FORWARD(Negative) { return -(*x[0]); }
FORWARD(AddConst) { return *x[0] + k_; }
FORWARD(SubtractConstL) { return k_ - *x[0]; }
FORWARD(SubtractConstR) { return *x[0] - k_; }
FORWARD(MultiplyConst) { return *x[0] * k_; }
FORWARD(DivideConstL) { return k_ / *x[0]; }
FORWARD(DivideConstR) { return *x[0] / k_; }
FORWARD(Transpose) { return T::transpose(*x[0]); }
FORWARD(Add) { return *x[0] + *x[1]; }
FORWARD(Subtract) { return *x[0] - *x[1]; }
FORWARD(Multiply) { return *x[0] * *x[1]; }
FORWARD(Divide) { return *x[0] / *x[1]; }
FORWARD(Dot) { return T::dot(*x[0], *x[1]); }
FORWARD(Exp) { return T::exp(*x[0]); }
FORWARD(Tanh) { return T::tanh(*x[0]); }
FORWARD(Sigmoid) { return T::sigmoid(*x[0]); }
FORWARD(ReLU) { return T::relu(*x[0]); }
FORWARD(Sum) { return T::sum(*x[0], dim_); }
FORWARD(LogSumExp) { return T::logsumexp(*x[0], dim_); }
FORWARD(Broadcast) { return T::broadcast(*x[0], dim_, size_); }
FORWARD(BatchSum) { return T::batch_sum(*x[0]); }
FORWARD(SoftmaxCrossEntropy) {
  return T::softmax_cross_entropy(*x[0], *x[1], dim_);
}

#undef FORWARD

#define BACKWARD(name) \
  void name::backward( \
      const Tensor &y, \
      const Tensor &yg, \
      const vector<const Tensor *> &x, \
      const vector<Tensor *> &xg) const
#define ADD(n, g) xg[n]->add_gradient(g)

BACKWARD(Positive) { ADD(0, yg); }
BACKWARD(Negative) { ADD(0, -yg); }
BACKWARD(AddConst) { ADD(0, yg); }
BACKWARD(SubtractConstL) { ADD(0, -yg); }
BACKWARD(SubtractConstR) { ADD(0, yg); }
BACKWARD(MultiplyConst) { ADD(0, k_ * yg); }
BACKWARD(DivideConstL) { ADD(0, -y * yg / *x[0]); }
BACKWARD(DivideConstR) { ADD(0, yg / k_); }
BACKWARD(Transpose) { ADD(0, T::transpose(yg)); }
BACKWARD(Add) { ADD(0, yg); ADD(1, yg); }
BACKWARD(Subtract) { ADD(0, yg); ADD(1, -yg); }
BACKWARD(Multiply) { ADD(0, *x[1] * yg); ADD(1, *x[0] * yg); }
BACKWARD(Divide) { Tensor a = yg / *x[1]; ADD(0, a); ADD(1, -a * y); }
BACKWARD(Dot) {
  ADD(0, T::dot(yg, T::transpose(*x[1])));
  ADD(1, T::dot(T::transpose(*x[0]), yg));
}
BACKWARD(Exp) { ADD(0, y * yg); }
BACKWARD(Tanh) { ADD(0, (1 - y * y) * yg); }
BACKWARD(Sigmoid) { ADD(0, y * (1 - y) * yg); }
BACKWARD(ReLU) { ADD(0, T::step(*x[0]) * yg); }
BACKWARD(Sum) { ADD(0, T::broadcast(yg, dim_, x[0]->shape()[dim_])); }
BACKWARD(LogSumExp) {
  // NOTE(odashi): dy/dx = softmax(x) = exp(x - y)
  const unsigned n = x[0]->shape()[dim_];
  ADD(0, T::exp(*x[0] - T::broadcast(y, dim_, n)) * T::broadcast(yg, dim_, n));
}
BACKWARD(Broadcast) { ADD(0, T::sum(yg, dim_)); }
BACKWARD(BatchSum) { ADD(0, yg); }
BACKWARD(SoftmaxCrossEntropy) {
  const Tensor log_softmax_x = T::log_softmax(*x[0], dim_);
  const Tensor bcast_yg = T::broadcast(yg, dim_, x[0]->shape()[dim_]);
  ADD(0, (T::exp(log_softmax_x) - *x[1]) * bcast_yg);
  ADD(1, -log_softmax_x * bcast_yg);
}

#undef BACKWARD
#undef ADD

}  // namespace functions
}  // namespace primitive
