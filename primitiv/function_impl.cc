#include <config.h>

#include <algorithm>
#include <primitiv/error.h>
#include <primitiv/function_impl.h>
#include <primitiv/parameter.h>
#include <primitiv/tensor_ops.h>
#include <primitiv/shape_ops.h>

using std::vector;

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

Shape Slice::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return shape_ops::slice(*args[0], dim_, lower_, upper_);
}

Tensor Slice::forward(const std::vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 1);
  return tensor_ops::slice(*args[0], dim_, lower_, upper_);
}

void Slice::backward(
    const Tensor &y, const Tensor &yg,
    const vector<const Tensor *> &x, const vector<Tensor *> &xg) const {
  xg[0]->add_gradient_offset(yg, dim_, lower_);
}

#define FWD_SHAPE_UNARY(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 1); \
    return *args[0]; \
  }

#define FWD_SHAPE_ARITHMETIC(clsname) \
  Shape clsname::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 2); \
    const Shape &a = *args[0]; \
    const Shape &b = *args[1]; \
    if (!a.is_compatible(b)) { \
      THROW_ERROR( \
          "Shape mismatched." \
          << " function: " << name() \
          << ", arg1: " << a.to_string() \
          << " != arg2: " << b.to_string()); \
    } \
    return a.resize_batch(std::max(a.batch_size(), b.batch_size())); \
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
  const Shape &a = *args[0];
  if (a.depth() > 2) {
    THROW_ERROR(
        "Shape mismatched."
        << " function: " << name()
        << ", arg1: " << a.to_string());
  }
  return Shape({a[1], a[0]}, a.batch_size());
}

Shape Dot::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 2);
  const Shape &a = *args[0];
  const Shape &b = *args[1];
  const unsigned a_bs = a.batch_size();
  const unsigned b_bs = b.batch_size();
  if (a.depth() > 2 || b.depth() > 2 || a[1] != b[0] ||
      !a.is_compatible_batch(b)) {
    THROW_ERROR(
        "Shape mismatched."
        << " function: " << name()
        << ", arg1: " << a.to_string()
        << " != arg2: " << b.to_string());
  }
  return Shape({a[0], b[1]}, std::max(a_bs, b_bs));
}

Shape BatchSum::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 1);
  return args[0]->resize_batch(1);
}

#undef FWD_SHAPE_UNARY
#undef FWD_SHAPE_ARITHMETIC

#define FORWARD(name) \
    Tensor name::forward(const vector<const Tensor *> &args) const

FORWARD(Positive) { return +(*args[0]); }
FORWARD(Negative) { return -(*args[0]); }
FORWARD(AddConst) { return *args[0] + k_; }
FORWARD(SubtractConstL) { return k_ - *args[0]; }
FORWARD(SubtractConstR) { return *args[0] - k_; }
FORWARD(MultiplyConst) { return *args[0] * k_; }
FORWARD(DivideConstL) { return k_ / *args[0]; }
FORWARD(DivideConstR) { return *args[0] / k_; }
FORWARD(Transpose) { return tensor_ops::transpose(*args[0]); }
FORWARD(Add) { return *args[0] + *args[1]; }
FORWARD(Subtract) { return *args[0] - *args[1]; }
FORWARD(Multiply) { return *args[0] * *args[1]; }
FORWARD(Divide) { return *args[0] / *args[1]; }
FORWARD(Dot) { return tensor_ops::dot(*args[0], *args[1]); }
FORWARD(Exp) { return tensor_ops::exp(*args[0]); }
FORWARD(Tanh) { return tensor_ops::tanh(*args[0]); }
FORWARD(Sigmoid) { return tensor_ops::sigmoid(*args[0]); }
FORWARD(ReLU) { return tensor_ops::relu(*args[0]); }
FORWARD(BatchSum) { return tensor_ops::batch_sum(*args[0]); }

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
BACKWARD(Transpose) { ADD(0, tensor_ops::transpose(yg)); }
BACKWARD(Add) { ADD(0, yg); ADD(1, yg); }
BACKWARD(Subtract) { ADD(0, yg); ADD(1, -yg); }
BACKWARD(Multiply) { ADD(0, *x[1] * yg); ADD(1, *x[0] * yg); }
BACKWARD(Divide) { Tensor a = yg / *x[1]; ADD(0, a); ADD(1, -a * y); }
BACKWARD(Dot) {
  ADD(0, tensor_ops::dot(yg, tensor_ops::transpose(*x[1])));
  ADD(1, tensor_ops::dot(tensor_ops::transpose(*x[0]), yg));
}
BACKWARD(Exp) { ADD(0, y * yg); }
BACKWARD(Tanh) { ADD(0, (1 - y * y) * yg); }
BACKWARD(Sigmoid) { ADD(0, y * (1 - y) * yg); }
BACKWARD(ReLU) { ADD(0, tensor_ops::step(*x[0]) * yg); }
BACKWARD(BatchSum) { ADD(0, yg); }

#undef BACKWARD
#undef ADD

}  // namespace functions
}  // namespace primitive
