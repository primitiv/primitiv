#ifndef PRIMITIV_FUNCTION_IMPL_H_
#define PRIMITIV_FUNCTION_IMPL_H_

#include <cstdint>
#include <primitiv/function.h>
#include <primitiv/parameter.h>
#include <primitiv/shape.h>

namespace primitiv {

class Device;

namespace functions {

#define DEFAULT_CLASS_DECL(name_) \
public: \
  Shape forward_shape(const std::vector<const Shape *> &args) const override; \
  Tensor forward(const std::vector<const Tensor *> &args) override; \
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
  Input(const Shape &shape, const std::vector<float> &data, Device &device);
  Device *get_device() const override { return &device_; }
  std::string name() const override { return "Input"; }
private:
  Shape shape_;
  std::vector<float> data_;
  Device &device_;
};

class ParameterInput : public primitiv::Function {
  NO_CTOR_CLASS_DECL(ParameterInput);
public:
  explicit ParameterInput(Parameter &param) : param_(param) {}
  Device *get_device() const override { return &param_.device(); }
  const Tensor *get_inner_value() const override { return &param_.value(); }
  std::string name() const override { return "ParameterInput"; }
private:
  primitiv::Parameter &param_;
};

class Copy : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Copy);
public:
  Copy(Device &device) : device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override { return "Copy"; }
private:
  Device &device_;
};

class Constant : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Constant);
public:
  Constant(const Shape &shape, float k, Device &device)
    : shape_(shape), k_(k), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return "Constant(" + std::to_string(k_) + ')';
  }
private:
  Shape shape_;
  float k_;
  Device &device_;
};

class IdentityMatrix : public primitiv::Function {
  NO_CTOR_CLASS_DECL(IdentityMatrix);
public:
  IdentityMatrix(std::uint32_t size, Device &device)
    : size_(size), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return "IdentityMatrix(" + std::to_string(size_) + ')';
  }
private:
  std::uint32_t size_;
  Device &device_;
};

class RandomBernoulli : public primitiv::Function {
  NO_CTOR_CLASS_DECL(RandomBernoulli);
public:
  RandomBernoulli(const Shape &shape, float p, Device &device)
    : shape_(shape), p_(p), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return "RandomBernoulli(" + std::to_string(p_) + ')';
  }
private:
  Shape shape_;
  float p_;
  Device &device_;
};

class RandomUniform : public primitiv::Function {
  NO_CTOR_CLASS_DECL(RandomUniform);
public:
  RandomUniform(const Shape &shape, float lower, float upper, Device &device)
    : shape_(shape), lower_(lower), upper_(upper), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return
      "RandomUniform(" + std::to_string(lower_) + ',' +
      std::to_string(upper_) + ')';
  }
private:
  Shape shape_;
  float lower_;
  float upper_;
  Device &device_;
};

class RandomNormal : public primitiv::Function {
  NO_CTOR_CLASS_DECL(RandomNormal);
public:
  RandomNormal(const Shape &shape, float mean, float sd, Device &device)
    : shape_(shape), mean_(mean), sd_(sd), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return
      "RandomNormal(" + std::to_string(mean_) + ',' +
      std::to_string(sd_) + ')';
  }
private:
  Shape shape_;
  float mean_;
  float sd_;
  Device &device_;
};

class RandomLogNormal : public primitiv::Function {
  NO_CTOR_CLASS_DECL(RandomLogNormal);
public:
  RandomLogNormal(const Shape &shape, float mean, float sd, Device &device)
    : shape_(shape), mean_(mean), sd_(sd), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return
      "RandomLogNormal(" + std::to_string(mean_) + ',' +
      std::to_string(sd_) + ')';
  }
private:
  Shape shape_;
  float mean_;
  float sd_;
  Device &device_;
};

class Pick : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Pick);
public:
  Pick(const std::vector<std::uint32_t> &ids, std::uint32_t dim)
    : ids_(ids), dim_(dim) {}
  std::string name() const override {
    return "Pick(" + std::to_string(dim_) + ')';
  };
private:
  std::vector<std::uint32_t> ids_;
  std::uint32_t dim_;
};

class Slice : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Slice);
public:
  Slice(std::uint32_t dim, std::uint32_t lower, std::uint32_t upper)
    : dim_(dim), lower_(lower), upper_(upper) {}
  std::string name() const override {
    return "Slice(" + std::to_string(dim_) +
      ',' + std::to_string(lower_) + ':' + std::to_string(upper_) + ')';
  }
private:
  std::uint32_t dim_;
  std::uint32_t lower_;
  std::uint32_t upper_;
};

class Concat : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Concat);
public:
  Concat(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "Concat(" + std::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class Reshape : public primitiv::Function {
  NO_CTOR_CLASS_DECL(Reshape);
public:
  explicit Reshape(const Shape &shape) : shape_(shape) {}
  std::string name() const override {
    return "Reshape(" + shape_.to_string() + ')';
  }
private:
  Shape shape_;
};

class Sum : public Function {
  NO_CTOR_CLASS_DECL(Sum);
public:
  explicit Sum(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "Sum(" + std::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class LogSumExp : public Function {
  NO_CTOR_CLASS_DECL(LogSumExp);
public:
  explicit LogSumExp(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "LogSumExp(" + std::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class Broadcast : public Function {
  NO_CTOR_CLASS_DECL(Broadcast);
public:
  Broadcast(std::uint32_t dim, std::uint32_t size) : dim_(dim), size_(size) {}
  std::string name() const override {
    return "Broadcast(" + std::to_string(dim_)
      + ',' + std::to_string(size_) + ')';
  }
private:
  std::uint32_t dim_;
  std::uint32_t size_;
};

class SoftmaxCrossEntropy : public Function {
  NO_CTOR_CLASS_DECL(SoftmaxCrossEntropy);
public:
  explicit SoftmaxCrossEntropy(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "SoftmaxCrossEntropy(" + std::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class SparseSoftmaxCrossEntropy : public Function {
  NO_CTOR_CLASS_DECL(SparseSoftmaxCrossEntropy);
public:
  explicit SparseSoftmaxCrossEntropy(
      const std::vector<std::uint32_t> ids, std::uint32_t dim) : ids_(ids), dim_(dim) {}
  std::string name() const override {
    return "SparseSoftmaxCrossEntropy(" + std::to_string(dim_) + ')';
  }
private:
  std::vector<std::uint32_t> ids_;
  std::uint32_t dim_;
  Tensor log_softmax_x_;  // Only used when PRIMITIV_USE_CACHE=ON
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

DECL_FUNC(Flatten);

DECL_FUNC(Positive);
DECL_FUNC(Negative);

DECL_FUNC_K(AddConst);
DECL_FUNC_K(SubtractConstR);
DECL_FUNC_K(SubtractConstL);
DECL_FUNC_K(MultiplyConst);
DECL_FUNC_K(DivideConstR);
DECL_FUNC_K(DivideConstL);
DECL_FUNC_K(PReLU);
DECL_FUNC_K(ELU);

DECL_FUNC(AddScalar);
DECL_FUNC(SubtractScalarR);
DECL_FUNC(SubtractScalarL);
DECL_FUNC(MultiplyScalar);
DECL_FUNC(DivideScalarR);
DECL_FUNC(DivideScalarL);

DECL_FUNC(Add);
DECL_FUNC(Subtract);
DECL_FUNC(Multiply);
DECL_FUNC(Divide);

DECL_FUNC(Transpose);
DECL_FUNC(MatrixMultiply);

DECL_FUNC(Sqrt);
DECL_FUNC(Exp);
DECL_FUNC(Log);
DECL_FUNC(Tanh);
DECL_FUNC(Sigmoid);
DECL_FUNC(Softplus);
DECL_FUNC(Sin);
DECL_FUNC(Cos);
DECL_FUNC(Tan);
DECL_FUNC(ReLU);
DECL_FUNC(LReLU);

DECL_FUNC(BatchSum);

#undef DECL_FUNC
#undef DECL_FUNC_K
#undef NO_CTOR_CLASS_DECL
#undef DEFAULT_CLASS_DECL

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_IMPL_H_
