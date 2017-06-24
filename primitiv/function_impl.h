#ifndef PRIMITIV_FUNCTION_IMPL_H_
#define PRIMITIV_FUNCTION_IMPL_H_

#include <primitiv/function.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;
class Parameter;

namespace functions {

#define DEFAULT_CLASS_DECL(name_) \
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

#define NO_CTOR_CLASS_DECL(name_) \
  DEFAULT_CLASS_DECL(name_); \
private: \
  name_() = delete;

class Input : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Input);
public:
  Input(const Shape &shape, Device *device, const std::vector<float> &data);
  std::string name() const override { return "Input"; }
private:
  Shape shape_;
  Device *device_;
  std::vector<float> data_;
};

class ParameterInput : public primitiv::Function {
  NO_CTOR_CLASS_DECL(ParameterInput);
public:
  explicit ParameterInput(Parameter *param) : param_(param) {}
  std::string name() const override { return "ParameterInput"; }
private:
  primitiv::Parameter *param_;
};

class Copy : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Copy);
public:
  Copy(Device *device) : device_(device) {}
  std::string name() const override { return "Copy"; }
private:
  Device *device_;
};

class RandomBernoulli : public primitiv::Function {
  NO_CTOR_CLASS_DECL(RandomBernoulli);
public:
  RandomBernoulli(const Shape &shape, float p, Device *device)
    : shape_(shape), p_(p), device_(device) {}
  std::string name() const override {
    return "RandomBernoulli(" + std::to_string(p_) + ')';
  }
private:
  Shape shape_;
  float p_;
  Device *device_;
};

class Pick : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Pick);
public:
  Pick(unsigned dim, const std::vector<unsigned> &ids)
    : dim_(dim), ids_(ids) {}
  std::string name() const override {
    return "Pick(" + std::to_string(dim_) + ')';
  };
private:
  unsigned dim_;
  std::vector<unsigned> ids_;
};

class Slice : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Slice);
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

class Concat : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Concat);
public:
  Concat(unsigned dim) : dim_(dim) {}
  std::string name() const override {
    return "Concat(" + std::to_string(dim_) + ')';
  }
private:
  unsigned dim_;
};

class Sum : public Function {
  NO_CTOR_CLASS_DECL(Sum);
public:
  explicit Sum(unsigned dim) : dim_(dim) {}
  std::string name() const override {
    return "Sum(" + std::to_string(dim_) + ')';
  }
private:
  unsigned dim_;
};

class LogSumExp : public Function {
  NO_CTOR_CLASS_DECL(LogSumExp);
public:
  explicit LogSumExp(unsigned dim) : dim_(dim) {}
  std::string name() const override {
    return "LogSumExp(" + std::to_string(dim_) + ')';
  }
private:
  unsigned dim_;
};

class Broadcast : public Function {
  NO_CTOR_CLASS_DECL(Broadcast);
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

class SoftmaxCrossEntropy : public Function {
  NO_CTOR_CLASS_DECL(SoftmaxCrossEntropy);
public:
  explicit SoftmaxCrossEntropy(unsigned dim) : dim_(dim) {}
  std::string name() const override {
    return "SoftmaxCrossEntropy(" + std::to_string(dim_) + ')';
  }
private:
  unsigned dim_;
};

// Function with no parameter.
#define DECL_FUNC(name_) \
  class name_ : public Function { \
    DEFAULT_CLASS_DECL(name_); \
  public: \
    name_() {} \
    std::string name() const override { return #name_; } \
  }

// Function with a constant.
#define DECL_FUNC_K(name_) \
  class name_ : public Function { \
    NO_CTOR_CLASS_DECL(name_); \
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
#undef NO_CTOR_CLASS_DECL
#undef DEFAULT_CLASS_DECL

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_IMPL_H_
