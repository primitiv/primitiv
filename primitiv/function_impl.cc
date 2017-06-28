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

Input::Input(const Shape &shape, const vector<float> &data, Device *device)
: shape_(shape)
, data_(data)
, device_(device) {
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
  return device_->new_tensor_by_vector(shape_, data_);
}

void Input::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
}

Shape ParameterInput::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return param_->shape();
}

Tensor ParameterInput::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return param_->value();
}

void ParameterInput::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  param_->add_gradient(cur_grad);
}

Shape Copy::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return *args[0];
}

Tensor Copy::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 1);
  return T::copy(*args[0], device_);
}

void Copy::backward(
    const Tensor &y, const Tensor &yg,
    const vector<const Tensor *> &x, const vector<Tensor *> &xg) const {
  xg[0]->add_gradient(T::copy(yg, xg[0]->device()));
}

Shape RandomBernoulli::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor RandomBernoulli::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return device_->random_bernoulli(shape_, p_);
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

Tensor RandomUniform::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return device_->random_uniform(shape_, lower_, upper_);
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

Tensor RandomNormal::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return device_->random_normal(shape_, mean_, sd_);
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

Tensor RandomLogNormal::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return device_->random_log_normal(shape_, mean_, sd_);
}

void RandomLogNormal::backward(
    const Tensor &, const Tensor &cur_grad,
    const vector<const Tensor *> &, const vector<Tensor *> &) const {
  // Nothing to do.
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
FWD_SHAPE_UNARY(Sqrt);
FWD_SHAPE_UNARY(Exp);
FWD_SHAPE_UNARY(Tanh);
FWD_SHAPE_UNARY(Sigmoid);
FWD_SHAPE_UNARY(Sin);
FWD_SHAPE_UNARY(Cos);
FWD_SHAPE_UNARY(Tan);
FWD_SHAPE_UNARY(ReLU);
FWD_SHAPE_UNARY(LReLU);
FWD_SHAPE_UNARY(PReLU);
FWD_SHAPE_SCALAR(AddScalar);
FWD_SHAPE_SCALAR(SubtractScalarR);
FWD_SHAPE_SCALAR(SubtractScalarL);
FWD_SHAPE_SCALAR(MultiplyScalar);
FWD_SHAPE_SCALAR(DivideScalarR);
FWD_SHAPE_SCALAR(DivideScalarL);
FWD_SHAPE_ELEMENTWISE(Add);
FWD_SHAPE_ELEMENTWISE(Subtract);
FWD_SHAPE_ELEMENTWISE(Multiply);
FWD_SHAPE_ELEMENTWISE(Divide);

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

#define FORWARD(name) \
    Tensor name::forward(const vector<const Tensor *> &x) const

FORWARD(Reshape) { return T::reshape(*x[0], shape_); }
FORWARD(Flatten) { return T::flatten(*x[0]); }

FORWARD(Positive) { return *x[0]; }
FORWARD(Negative) { return -(*x[0]); }

FORWARD(AddConst) { return *x[0] + k_; }
FORWARD(SubtractConstR) { return *x[0] - k_; }
FORWARD(SubtractConstL) { return k_ - *x[0]; }
FORWARD(MultiplyConst) { return *x[0] * k_; }
FORWARD(DivideConstR) { return *x[0] / k_; }
FORWARD(DivideConstL) { return k_ / *x[0]; }

FORWARD(AddScalar) { return *x[0] + *x[1]; }
FORWARD(SubtractScalarR) { return *x[0] - *x[1]; }
FORWARD(SubtractScalarL) { return *x[1] - *x[0]; }
FORWARD(MultiplyScalar) { return *x[0] * *x[1]; }
FORWARD(DivideScalarR) { return *x[0] / *x[1]; }
FORWARD(DivideScalarL) { return *x[1] / *x[0]; }

FORWARD(Add) { return *x[0] + *x[1]; }
FORWARD(Subtract) { return *x[0] - *x[1]; }
FORWARD(Multiply) { return *x[0] * *x[1]; }
FORWARD(Divide) { return *x[0] / *x[1]; }

FORWARD(Transpose) { return T::transpose(*x[0]); }
FORWARD(MatrixMultiply) { return T::matmul(*x[0], *x[1]); }

FORWARD(Sqrt) { return T::sqrt(*x[0]); }
FORWARD(Exp) { return T::exp(*x[0]); }
FORWARD(Tanh) { return T::tanh(*x[0]); }
FORWARD(Sigmoid) { return T::sigmoid(*x[0]); }
FORWARD(Sin) { return T::sin(*x[0]); }
FORWARD(Cos) { return T::cos(*x[0]); }
FORWARD(Tan) { return T::tan(*x[0]); }
FORWARD(ReLU) { return T::relu(*x[0]); }
FORWARD(LReLU) { return T::lrelu(*x[0]); }
FORWARD(PReLU) { return T::prelu(*x[0], k_); }

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

BACKWARD(Reshape) { ADD(0, yg.reshape(x[0]->shape())); }
BACKWARD(Flatten) { ADD(0, yg.reshape(x[0]->shape())); }

BACKWARD(Positive) { ADD(0, yg); }
BACKWARD(Negative) { ADD(0, -yg); }

BACKWARD(AddConst) { ADD(0, yg); }
BACKWARD(SubtractConstR) { ADD(0, yg); }
BACKWARD(SubtractConstL) { ADD(0, -yg); }
BACKWARD(MultiplyConst) { ADD(0, k_ * yg); }
BACKWARD(DivideConstR) { ADD(0, yg / k_); }
BACKWARD(DivideConstL) { ADD(0, -y * yg / *x[0]); }

BACKWARD(AddScalar) {
  ADD(0, yg);
  ADD(1, T::sum(yg.flatten(), 0));
}
BACKWARD(SubtractScalarR) {
  ADD(0, yg);
  ADD(1, -T::sum(yg.flatten(), 0));
}
BACKWARD(SubtractScalarL) {
  ADD(0, -yg);
  ADD(1, T::sum(yg.flatten(), 0));
}
BACKWARD(MultiplyScalar) {
  ADD(0, *x[1] * yg);
  ADD(1, T::sum((*x[0] * yg).flatten(), 0));
}
BACKWARD(DivideScalarR) {
  const Tensor a = yg / *x[1];
  ADD(0, a);
  ADD(1, T::sum((-a * y).flatten(), 0));
}
BACKWARD(DivideScalarL) {
  const Tensor a = yg / *x[0];
  ADD(0, -a * y);
  ADD(1, T::sum(a.flatten(), 0));
}

BACKWARD(Add) {
  ADD(0, yg);
  ADD(1, yg);
}
BACKWARD(Subtract) {
  ADD(0, yg);
  ADD(1, -yg);
}
BACKWARD(Multiply) {
  ADD(0, *x[1] * yg);
  ADD(1, *x[0] * yg);
}
BACKWARD(Divide) {
  const Tensor a = yg / *x[1];
  ADD(0, a);
  ADD(1, -a * y);
}

BACKWARD(Transpose) { ADD(0, T::transpose(yg)); }
BACKWARD(MatrixMultiply) {
  // TODO(odashi): This requires large memory. Suppress it.
  ADD(0, T::matmul(yg, T::transpose(*x[1])));
  ADD(1, T::matmul(T::transpose(*x[0]), yg));
}

BACKWARD(Sqrt) { ADD(0, .5 * yg / y); }
BACKWARD(Exp) { ADD(0, y * yg); }
BACKWARD(Tanh) { ADD(0, (1 - y * y) * yg); }
BACKWARD(Sigmoid) { ADD(0, y * (1 - y) * yg); }
BACKWARD(Sin) { ADD(0, T::cos(*x[0]) * yg); }
BACKWARD(Cos) { ADD(0, -T::sin(*x[0]) * yg); }
BACKWARD(Tan) { ADD(0, (1 + y * y) * yg); }
BACKWARD(ReLU) { ADD(0, T::step(*x[0]) * yg); }
BACKWARD(LReLU) { ADD(0, T::lstep(*x[0]) * yg); }
BACKWARD(PReLU) { ADD(0, T::pstep(*x[0], k_) * yg); }

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
