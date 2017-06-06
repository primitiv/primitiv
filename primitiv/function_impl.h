#ifndef PRIMITIV_FUNCTION_IMPL_H_
#define PRIMITIV_FUNCTION_IMPL_H_

#include <primitiv/function.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;
class Parameter;

namespace functions {

/**
 * Function object that behaves data source of the computation graph.
 */
class Input : public primitiv::Function {
  Input() = delete;
  Input(const Input &) = delete;
  Input(Input &&) = delete;
  Input &operator=(const Input &) = delete;
  Input &operator=(Input &&) = delete;

public:
  Input(const Shape &shape, Device *device, const std::vector<float> &data);
  ~Input() override = default;
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

/**
 * Function object to manage parameters.
 */
class ParameterInput : public primitiv::Function {
  ParameterInput() = delete;
  ParameterInput(const ParameterInput &) = delete;
  ParameterInput(ParameterInput &&) = delete;
  ParameterInput &operator=(const ParameterInput &) = delete;
  ParameterInput &operator=(ParameterInput &&) = delete;

public:
  ParameterInput(Parameter *param) : param_(param) {}
  ~ParameterInput() override = default;
  Shape forward_shape(const std::vector<const Shape *> &args) const override;
  Tensor forward(const std::vector<const Tensor *> &args) const override;
  void backward(
      const Tensor &cur_value,
      const Tensor &cur_grad,
      const std::vector<const Tensor *> &arg_values,
      const std::vector<Tensor *> &arg_grads) const override;
  inline std::string name() const override { return "ParameterInput"; }

private:
  primitiv::Parameter *param_;
};

// Function with no parameter.
#define DECL_FUNC(name_) \
  class name_ : public Function { \
    name_(const name_ &) = delete; \
    name_(name_ &&) = delete; \
    name_ &operator=(const name_ &) = delete; \
    name_ &operator=(name_ &&) = delete; \
  public: \
    inline name_() {} \
    ~name_() override = default; \
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

// Function with a constant.
#define DECL_FUNC_K(name_) \
  class name_ : public Function { \
    name_() = delete; \
    name_(const name_ &) = delete; \
    name_(name_ &&) = delete; \
    name_ &operator=(const name_ &) = delete; \
    name_ &operator=(name_ &&) = delete; \
  public: \
    inline name_(const float k) : k_(k) {} \
    ~name_() override = default; \
    Shape forward_shape( \
        const std::vector<const Shape *> &args) const override; \
    Tensor forward(const std::vector<const Tensor *> &args) const override; \
    void backward( \
      const Tensor &cur_value, \
      const Tensor &cur_grad, \
      const std::vector<const Tensor *> &arg_values, \
      const std::vector<Tensor *> &arg_grads) const override; \
    inline std::string name() const override { \
      return std::string(#name_) + '(' + std::to_string(k_) + ')'; \
    } \
  private: \
    float k_; \
  }

DECL_FUNC(Positive);
DECL_FUNC(Negative);
DECL_FUNC(Add);
DECL_FUNC(Subtract);
DECL_FUNC(Multiply);
DECL_FUNC(Divide);
DECL_FUNC(Transpose);
DECL_FUNC(Dot);
DECL_FUNC(Exp);
DECL_FUNC(Tanh);
DECL_FUNC(Sigmoid);
DECL_FUNC(ReLU);
DECL_FUNC(BatchSum);

DECL_FUNC_K(AddConst);
DECL_FUNC_K(SubtractConstL);
DECL_FUNC_K(SubtractConstR);
DECL_FUNC_K(MultiplyConst);
DECL_FUNC_K(DivideConstL);
DECL_FUNC_K(DivideConstR);

#undef DECL_FUNC
#undef DECL_FUNC_K

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_IMPL_H_
