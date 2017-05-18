#include <config.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <primitiv/function_impl.h>
#include <primitiv/tensor_ops.h>

using std::vector;

namespace primitiv {
namespace functions {

#define CHECK_ARGNUM(args, n) \
  if (args.size() != n) { \
    std::stringstream ss; \
    ss << "Number of arguments mismatched." \
       << " function: " << name() \
       << ", required: " << n \
       << " != actual: " << args.size(); \
    throw std::runtime_error(ss.str()); \
  }

#define RETURN_ADD_SHAPE(a, b) { \
  const unsigned a_bs = (a).batch_size(); \
  const unsigned b_bs = (b).batch_size(); \
  if ((a).dims() != (b).dims() || (a_bs != b_bs && a_bs > 1 && b_bs > 1)) { \
    std::stringstream ss; \
    ss << "Shape mismatched." \
       << " function: " << name() \
       << ", arg1: " << (a).to_string() \
       << " != arg2: " << (b).to_string(); \
    throw std::runtime_error(ss.str()); \
  } \
  return Shape((a).dims(), std::max(a_bs, b_bs)); \
}

Input::Input(const Shape &shape, Device *device, const vector<float> &data)
: shape_(shape)
, device_(device)
, data_(data) {
  const unsigned shape_size = shape_.size();
  if (data_.size() != shape_size) {
    std::stringstream ss;
    ss << "Data sizes mismatched."
       << " function: Input"
       << ", required: " << shape_size << " (" << shape_.to_string() << ")"
       << ", actual: " << data_.size();
    throw std::runtime_error(ss.str());
  }
}

Shape Input::forward_shape(const vector<const Shape *> &args) const {
  CHECK_ARGNUM(args, 0);
  return shape_;
}

Tensor Input::forward(const vector<const Tensor *> &args) const {
  CHECK_ARGNUM(args, 0);
  return Tensor(shape_, device_, data_);
}

#define FWD_SHAPE_ARITHMETIC_TT(name) \
  Shape name::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 2); \
    RETURN_ADD_SHAPE(*args[0], *args[1]); \
  }
FWD_SHAPE_ARITHMETIC_TT(Add);
FWD_SHAPE_ARITHMETIC_TT(Subtract);
FWD_SHAPE_ARITHMETIC_TT(Multiply);
FWD_SHAPE_ARITHMETIC_TT(Divide);
#undef FWD_SHAPE_ARITHMETIC_TT

#define FWD_SHAPE_ARITHMETIC_TC(name) \
  Shape name::forward_shape(const vector<const Shape *> &args) const { \
    CHECK_ARGNUM(args, 1); \
    return *args[0]; \
  }
FWD_SHAPE_ARITHMETIC_TC(AddConst)
FWD_SHAPE_ARITHMETIC_TC(SubtractConstL)
FWD_SHAPE_ARITHMETIC_TC(SubtractConstR)
FWD_SHAPE_ARITHMETIC_TC(MultiplyConst)
FWD_SHAPE_ARITHMETIC_TC(DivideConstL)
FWD_SHAPE_ARITHMETIC_TC(DivideConstR)
#undef FWD_SHAPE_ARITHMETIC_TC

#define FORWARD(name) \
    Tensor name::forward(const vector<const Tensor *> &args) const
FORWARD(Add) { return *args[0] + *args[1]; }
FORWARD(Subtract) { return *args[0] - *args[1]; }
FORWARD(Multiply) { return *args[0] * *args[1]; }
FORWARD(Divide) { return *args[0] / *args[1]; }
FORWARD(AddConst) { return *args[0] + k_; }
FORWARD(SubtractConstL) { return k_ - *args[0]; }
FORWARD(SubtractConstR) { return *args[0] - k_; }
FORWARD(MultiplyConst) { return *args[0] * k_; }
FORWARD(DivideConstL) { return k_ / *args[0]; }
FORWARD(DivideConstR) { return *args[0] / k_; }
#undef FORWARD

#define BACKWARD(name) \
    void name::backward( \
        const Tensor &cur_value, \
        const Tensor &cur_grad, \
        const vector<const Tensor *> &arg_values, \
        const vector<Tensor *> &arg_grads) const
BACKWARD(Add) {
  arg_grads[0]->add_gradient(cur_grad);
  arg_grads[1]->add_gradient(cur_grad);
}
BACKWARD(Subtract) {
  arg_grads[0]->add_gradient(cur_grad);
  arg_grads[1]->add_gradient(-cur_grad);
}
BACKWARD(Multiply) {
  arg_grads[0]->add_gradient(*arg_values[1] * cur_grad);
  arg_grads[1]->add_gradient(*arg_values[0] * cur_grad);
}
BACKWARD(Divide) { throw std::runtime_error("Divide: not implemented"); }
BACKWARD(AddConst) {
  arg_grads[0]->add_gradient(cur_grad);
}
BACKWARD(SubtractConstL) { throw std::runtime_error("SubtractConstL: not implemented"); }
BACKWARD(SubtractConstR) { throw std::runtime_error("SubtractConstR: not implemented"); }
BACKWARD(MultiplyConst) { throw std::runtime_error("MultiplyConst: not implemented"); }
BACKWARD(DivideConstL) { throw std::runtime_error("DivideConstL: not implemented"); }
BACKWARD(DivideConstR) { throw std::runtime_error("DivideConstR: not implemented"); }
#undef BACKWARD

}  // namespace functions
}  // namespace primitive
