#ifndef PRIMITIV_OPERATOR_IMPL_H_
#define PRIMITIV_OPERATOR_IMPL_H_

#include <cstdint>
#include <primitiv/operator.h>
#include <primitiv/parameter.h>
#include <primitiv/shape.h>
#include <primitiv/string_utils.h>

namespace primitiv {

class Device;

namespace operators {

#define PRIMITIV_DEFAULT_CLASS_DECL(name_) \
public: \
  Shape forward_shape(const std::vector<const Shape *> &args) const override; \
  Tensor forward(const std::vector<const Tensor *> &args) override; \
  void backward( \
      const Tensor &cur_value, \
      const Tensor &cur_grad, \
      const std::vector<const Tensor *> &arg_values, \
      const std::vector<Tensor *> &arg_grads) const override;

#define PRIMITIV_NO_CTOR_CLASS_DECL(name_) \
  PRIMITIV_DEFAULT_CLASS_DECL(name_); \
private: \
  name_() = delete;

class Input : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Input);
public:
  Input(const Shape &shape, const std::vector<float> &data, Device &device);
  Device *get_device() const override { return &device_; }
  std::string name() const override { return "Input"; }
private:
  Shape shape_;
  std::vector<float> data_;
  Device &device_;
};

class ParameterInput : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(ParameterInput);
public:
  explicit ParameterInput(Parameter &param) : param_(param) {}
  Device *get_device() const override { return &param_.device(); }
  const Tensor *get_inner_value() const override { return &param_.value(); }
  std::string name() const override { return "ParameterInput"; }
private:
  primitiv::Parameter &param_;
};

class Copy : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Copy);
public:
  Copy(Device &device) : device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override { return "Copy"; }
private:
  Device &device_;
};

class Constant : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Constant);
public:
  Constant(const Shape &shape, float k, Device &device)
    : shape_(shape), k_(k), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return "Constant(" + string_utils::to_string(k_) + ')';
  }
private:
  Shape shape_;
  float k_;
  Device &device_;
};

class IdentityMatrix : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(IdentityMatrix);
public:
  IdentityMatrix(std::uint32_t size, Device &device)
    : size_(size), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return "IdentityMatrix(" + string_utils::to_string(size_) + ')';
  }
private:
  std::uint32_t size_;
  Device &device_;
};

class RandomBernoulli : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(RandomBernoulli);
public:
  RandomBernoulli(const Shape &shape, float p, Device &device)
    : shape_(shape), p_(p), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return "RandomBernoulli(" + string_utils::to_string(p_) + ')';
  }
private:
  Shape shape_;
  float p_;
  Device &device_;
};

class RandomUniform : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(RandomUniform);
public:
  RandomUniform(const Shape &shape, float lower, float upper, Device &device)
    : shape_(shape), lower_(lower), upper_(upper), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return
      "RandomUniform(" + string_utils::to_string(lower_) + ',' +
      string_utils::to_string(upper_) + ')';
  }
private:
  Shape shape_;
  float lower_;
  float upper_;
  Device &device_;
};

class RandomNormal : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(RandomNormal);
public:
  RandomNormal(const Shape &shape, float mean, float sd, Device &device)
    : shape_(shape), mean_(mean), sd_(sd), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return
      "RandomNormal(" + string_utils::to_string(mean_) + ',' +
      string_utils::to_string(sd_) + ')';
  }
private:
  Shape shape_;
  float mean_;
  float sd_;
  Device &device_;
};

class RandomLogNormal : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(RandomLogNormal);
public:
  RandomLogNormal(const Shape &shape, float mean, float sd, Device &device)
    : shape_(shape), mean_(mean), sd_(sd), device_(device) {}
  Device *get_device() const override { return &device_; }
  std::string name() const override {
    return
      "RandomLogNormal(" + string_utils::to_string(mean_) + ',' +
      string_utils::to_string(sd_) + ')';
  }
private:
  Shape shape_;
  float mean_;
  float sd_;
  Device &device_;
};

class Pick : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Pick);
public:
  Pick(const std::vector<std::uint32_t> &ids, std::uint32_t dim)
    : ids_(ids), dim_(dim) {}
  std::string name() const override {
    return "Pick(" + string_utils::to_string(dim_) + ')';
  };
private:
  std::vector<std::uint32_t> ids_;
  std::uint32_t dim_;
};

class Slice : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Slice);
public:
  Slice(std::uint32_t dim, std::uint32_t lower, std::uint32_t upper)
    : dim_(dim), lower_(lower), upper_(upper) {}
  std::string name() const override {
    return "Slice(" + string_utils::to_string(dim_) +
      ',' + string_utils::to_string(lower_) + ':' + string_utils::to_string(upper_) + ')';
  }
private:
  std::uint32_t dim_;
  std::uint32_t lower_;
  std::uint32_t upper_;
};

class Concat : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Concat);
public:
  Concat(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "Concat(" + string_utils::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class Reshape : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Reshape);
public:
  explicit Reshape(const Shape &shape) : shape_(shape) {}
  std::string name() const override {
    return "Reshape(" + shape_.to_string() + ')';
  }
private:
  Shape shape_;
};

class Sum : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Sum);
public:
  explicit Sum(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "Sum(" + string_utils::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class LogSumExp : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(LogSumExp);
public:
  explicit LogSumExp(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "LogSumExp(" + string_utils::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class Broadcast : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Broadcast);
public:
  Broadcast(std::uint32_t dim, std::uint32_t size) : dim_(dim), size_(size) {}
  std::string name() const override {
    return "Broadcast(" + string_utils::to_string(dim_)
      + ',' + string_utils::to_string(size_) + ')';
  }
private:
  std::uint32_t dim_;
  std::uint32_t size_;
};

class SoftmaxCrossEntropy : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(SoftmaxCrossEntropy);
public:
  explicit SoftmaxCrossEntropy(std::uint32_t dim) : dim_(dim) {}
  std::string name() const override {
    return "SoftmaxCrossEntropy(" + string_utils::to_string(dim_) + ')';
  }
private:
  std::uint32_t dim_;
};

class SparseSoftmaxCrossEntropy : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(SparseSoftmaxCrossEntropy);
public:
  explicit SparseSoftmaxCrossEntropy(
      const std::vector<std::uint32_t> ids, std::uint32_t dim) : ids_(ids), dim_(dim) {}
  std::string name() const override {
    return "SparseSoftmaxCrossEntropy(" + string_utils::to_string(dim_) + ')';
  }
private:
  std::vector<std::uint32_t> ids_;
  std::uint32_t dim_;
  Tensor log_softmax_x_;  // Only used when PRIMITIV_USE_CACHE=ON
};

class StopGradient : public Operator {
  PRIMITIV_DEFAULT_CLASS_DECL(StopGradient);
public:
  StopGradient() {}
  std::string name() const override { return "StopGradient"; }
};

// Operator with no parameter.
#define PRIMITIV_DECL_OPERATOR(name_) \
  class name_ : public Operator { \
    PRIMITIV_DEFAULT_CLASS_DECL(name_); \
  public: \
    name_() {} \
    std::string name() const override { return #name_; } \
  }

// Operator with a constant.
#define PRIMITIV_DECL_OPERATOR_K(name_) \
  class name_ : public Operator { \
    PRIMITIV_NO_CTOR_CLASS_DECL(name_); \
  public: \
    explicit name_(float k) : k_(k) {} \
    std::string name() const override { \
      return #name_"(" + string_utils::to_string(k_) + ')'; \
    } \
  private: \
    float k_; \
  }

PRIMITIV_DECL_OPERATOR(Flatten);

PRIMITIV_DECL_OPERATOR(Positive);
PRIMITIV_DECL_OPERATOR(Negative);

PRIMITIV_DECL_OPERATOR_K(AddConst);
PRIMITIV_DECL_OPERATOR_K(SubtractConstR);
PRIMITIV_DECL_OPERATOR_K(SubtractConstL);
PRIMITIV_DECL_OPERATOR_K(MultiplyConst);
PRIMITIV_DECL_OPERATOR_K(DivideConstR);
PRIMITIV_DECL_OPERATOR_K(DivideConstL);
PRIMITIV_DECL_OPERATOR_K(PowConstR);
PRIMITIV_DECL_OPERATOR_K(PowConstL);
PRIMITIV_DECL_OPERATOR_K(PReLU);
PRIMITIV_DECL_OPERATOR_K(ELU);

class PowN : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(PowN);
public:
  explicit PowN(std::int32_t k) : k_(k) {}
  std::string name() const override {
    return "PowN(" + string_utils::to_string(k_) + ')';
  }
private:
  std::int32_t k_;
};

PRIMITIV_DECL_OPERATOR(AddScalar);
PRIMITIV_DECL_OPERATOR(SubtractScalarR);
PRIMITIV_DECL_OPERATOR(SubtractScalarL);
PRIMITIV_DECL_OPERATOR(MultiplyScalar);
PRIMITIV_DECL_OPERATOR(DivideScalarR);
PRIMITIV_DECL_OPERATOR(DivideScalarL);
PRIMITIV_DECL_OPERATOR(PowScalarR);
PRIMITIV_DECL_OPERATOR(PowScalarL);

PRIMITIV_DECL_OPERATOR(Add);
PRIMITIV_DECL_OPERATOR(Subtract);
PRIMITIV_DECL_OPERATOR(Multiply);
PRIMITIV_DECL_OPERATOR(Divide);
PRIMITIV_DECL_OPERATOR(Pow);

PRIMITIV_DECL_OPERATOR(Transpose);
PRIMITIV_DECL_OPERATOR(MatrixMultiply);

PRIMITIV_DECL_OPERATOR(Sqrt);
PRIMITIV_DECL_OPERATOR(Exp);
PRIMITIV_DECL_OPERATOR(Log);
PRIMITIV_DECL_OPERATOR(Tanh);
PRIMITIV_DECL_OPERATOR(Sigmoid);
PRIMITIV_DECL_OPERATOR(Softplus);
PRIMITIV_DECL_OPERATOR(Sin);
PRIMITIV_DECL_OPERATOR(Cos);
PRIMITIV_DECL_OPERATOR(Tan);
PRIMITIV_DECL_OPERATOR(ReLU);
PRIMITIV_DECL_OPERATOR(LReLU);

PRIMITIV_DECL_OPERATOR(BatchSum);

class Convolution2D : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(Convolution2D);

public:
  Convolution2D(
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1)
  : padding0_(padding0), padding1_(padding1)
  , stride0_(stride0), stride1_(stride1)
  , dilation0_(dilation0), dilation1_(dilation1) {}

  std::string name() const override {
    return "Convolution2D("
      + string_utils::to_string(padding0_) + ","
      + string_utils::to_string(padding1_) + ","
      + string_utils::to_string(stride0_) + ","
      + string_utils::to_string(stride1_) + ","
      + string_utils::to_string(dilation0_) + ","
      + string_utils::to_string(dilation1_) + ")";
  }

private:
  std::uint32_t padding0_, padding1_;
  std::uint32_t stride0_, stride1_;
  std::uint32_t dilation0_, dilation1_;
};

class MaxPooling2D : public Operator {
  PRIMITIV_NO_CTOR_CLASS_DECL(MaxPooling2D);

public:
  MaxPooling2D(
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1)
  : window0_(window0), window1_(window1)
  , padding0_(padding0), padding1_(padding1)
  , stride0_(stride0), stride1_(stride1) {}

  std::string name() const override {
    return "MaxPooling2D("
      + string_utils::to_string(window0_) + ","
      + string_utils::to_string(window1_) + ","
      + string_utils::to_string(padding0_) + ","
      + string_utils::to_string(padding1_) + ","
      + string_utils::to_string(stride0_) + ","
      + string_utils::to_string(stride1_) + ")";
  }

private:
  std::uint32_t window0_, window1_;
  std::uint32_t padding0_, padding1_;
  std::uint32_t stride0_, stride1_;
};

#undef PRIMITIV_DECL_OPERATOR
#undef PRIMITIV_DECL_OPERATOR_K
#undef PRIMITIV_NO_CTOR_CLASS_DECL
#undef PRIMITIV_DEFAULT_CLASS_DECL

}  // namespace operators
}  // namespace primitiv

#endif  // PRIMITIV_OPERATOR_IMPL_H_
