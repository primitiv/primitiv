#ifndef PRIMITIV_FUNCTION_IMPL_H_
#define PRIMITIV_FUNCTION_IMPL_H_

#include <primitiv/function.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;
class Parameter;

namespace functions {

#define DEFAULT_METHODS(name_) \
private: \
  name_(const name_ &) = delete; \
  name_(name_ &&) = delete; \
  name_ &operator=(const name_ &) = delete; \
  name_ &operator=(name_ &&) = delete; \
public: \
  Shape forward_shape(const std::vector<const Shape *> &args) const override; \
  Tensor forward(const std::vector<const Tensor *> &args) const override; \
  void backward( \
      const Tensor &cur_value, \
      const Tensor &cur_grad, \
      const std::vector<const Tensor *> &arg_values, \
      const std::vector<Tensor *> &arg_grads) const override;

/**
 * Function object that behaves data source of the computation graph.
 */
class Input : public primitiv::Function {
  DEFAULT_METHODS(Input);

private:
  Input() = delete;

public:
  Input(const Shape &shape, Device *device, const std::vector<float> &data);
  std::string name() const override { return "Input"; }

private:
  Shape shape_;
  Device *device_;
  std::vector<float> data_;
};

/**
 * Function object to manage parameters.
 */
class ParameterInput : public primitiv::Function {
  DEFAULT_METHODS(ParameterInput);

private:
  ParameterInput() = delete;

public:
  explicit ParameterInput(Parameter *param) : param_(param) {}
  std::string name() const override { return "ParameterInput"; }

private:
  primitiv::Parameter *param_;
};

// Function to slice a tensor.
class Slice : public primitiv::Function {
  DEFAULT_METHODS(Slice);

private:
  Slice() = delete;

public:
  Slice(unsigned dim, unsigned lower, unsigned upper)
  : dim_(dim), lower_(lower), upper_(upper) {}
  std::string name() const override {
    return "Slice(" + std::to_string(dim_) +
      ',' + std::to_string(lower_) + ':' + std::to_string(upper_) + ')';
  }

private:
  unsigned dim_;
  unsigned lower_;
  unsigned upper_;
};

// Function to concat tensors.
class Concat : public primitiv::Function {
  DEFAULT_METHODS(Concat);

private:
  Concat() = delete;

public:
  Concat(unsigned dim) : dim_(dim) {}
  std::string name() const override {
    return "Concat(" + std::to_string(dim_) + ')';
  }

private:
  unsigned dim_;
};

/**
 * Function to sum a dimension.
 */
class Sum : public Function {
  DEFAULT_METHODS(Sum);

private:
  Sum() = delete;

public:
  explicit Sum(unsigned dim) : dim_(dim) {}
  std::string name() const override {
    return "Sum(" + std::to_string(dim_) + ')';
  }

private:
  unsigned dim_;
};

/**
 * Function to broadcast a dimension.
 */
class Broadcast : public Function {
  DEFAULT_METHODS(Broadcast);

private:
  Broadcast() = delete;

public:
  Broadcast(unsigned dim, unsigned size) : dim_(dim), size_(size) {}
  std::string name() const override {
    return "Broadcast(" + std::to_string(dim_)
      + ',' + std::to_string(size_) + ')';
  }

private:
  unsigned dim_;
  unsigned size_;
};

// Function with no parameter.
#define DECL_FUNC(name_) \
  class name_ : public Function { \
    DEFAULT_METHODS(name_); \
  public: \
    name_() {} \
    std::string name() const override { return #name_; } \
  }

// Function with a constant.
#define DECL_FUNC_K(name_) \
  class name_ : public Function { \
    DEFAULT_METHODS(name_); \
  private: \
    name_() = delete; \
  public: \
    explicit name_(float  k) : k_(k) {} \
    std::string name() const override { \
      return #name_"(" + std::to_string(k_) + ')'; \
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
#undef DEFAULT_METHODS

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_IMPL_H_
