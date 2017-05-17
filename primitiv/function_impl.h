#ifndef PRIMITIV_FUNCTION_IMPL_H_
#define PRIMITIV_FUNCTION_IMPL_H_

#include <primitiv/function.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;

namespace functions {

/**
 * Function object that behaves data source of the computation graph.
 */
class Input : public primitiv::Function {
  Input() = delete;
public:
  Input(const Shape &shape, Device *device, const std::vector<float> &data);
  Shape forward_shape(const std::vector<const Shape *> &args) const override;
  Tensor forward(const std::vector<const Tensor *> &args) const override;
  inline void backward(
      const Tensor &cur_value,
      const Tensor &cur_grad,
      const std::vector<const Tensor *> &arg_values,
      const std::vector<Tensor *> &arg_grads) const override {}
  inline std::string name() const override { return "Input"; }
private:
  Shape shape_;
  Device *device_;
  std::vector<float> data_;
};

// Arithmetic functions between two tensors.
#define DECL_ARITHMETIC_TT(name_) \
  class name_ : public Function { \
  public: \
    Shape forward_shape( \
        const std::vector<const Shape *> &args) const override; \
    Tensor forward(const std::vector<const Tensor *> &args) const override; \
    void backward( \
      const Tensor &cur_value, \
      const Tensor &cur_grad, \
      const std::vector<const Tensor *> &arg_values, \
      const std::vector<Tensor *> &arg_grads) const override; \
    inline std::string name() const override { return #name_; } \
  }
DECL_ARITHMETIC_TT(Add);
DECL_ARITHMETIC_TT(Subtract);
DECL_ARITHMETIC_TT(Multiply);
DECL_ARITHMETIC_TT(Divide);
#undef DECL_ARITHMETIC_TT

// Arithmetic functions between a tensor and a constant.
#define DECL_ARITHMETIC_TC(name_) \
  class name_ : public Function { \
    name_() = delete; \
  public: \
    inline name_(const float k) : k_(k) {} \
    Shape forward_shape( \
        const std::vector<const Shape *> &args) const override; \
    Tensor forward(const std::vector<const Tensor *> &args) const override; \
    void backward( \
      const Tensor &cur_value, \
      const Tensor &cur_grad, \
      const std::vector<const Tensor *> &arg_values, \
      const std::vector<Tensor *> &arg_grads) const override; \
    inline std::string name() const override { return #name_; } \
  private: \
    float k_; \
  }
DECL_ARITHMETIC_TC(AddConst);
DECL_ARITHMETIC_TC(SubtractConstL);
DECL_ARITHMETIC_TC(SubtractConstR);
DECL_ARITHMETIC_TC(MultiplyConst);
DECL_ARITHMETIC_TC(DivideConstL);
DECL_ARITHMETIC_TC(DivideConstR);
#undef DECL_ARITHMETIC_TC

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_IMPL_H_
