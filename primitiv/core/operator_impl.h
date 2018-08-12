#ifndef PRIMITIV_CORE_OPERATOR_IMPL_H_
#define PRIMITIV_CORE_OPERATOR_IMPL_H_

#include <cstdint>

#include <primitiv/core/operator.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/shape.h>

namespace primitiv {

class Device;

namespace operators {

#define PRIMITIV_DECL_DEFAULTS(argn, retn, inval) \
public: \
  std::string name() const override; \
  std::uint32_t num_arguments() const override { return argn; }; \
  std::uint32_t num_returns() const override { return retn; }; \
  bool has_inner_values() const override { return inval; }; \
  void forward_shape( \
      const std::vector<const Shape *> &args, \
      const std::vector<Shape *> &rets) const override; \
  void backward( \
      const std::vector<const Tensor *> &args_v, \
      const std::vector<const Tensor *> &rets_v, \
      const std::vector<const Tensor *> &rets_g, \
      const std::vector<Tensor *> &args_g) const override;

#define PRIMITIV_DECL_DEFAULTS_AND_FORWARD(argn, retn) \
  PRIMITIV_DECL_DEFAULTS(argn, retn, false) \
public: \
  void forward( \
      const std::vector<const Tensor *> &args, \
      const std::vector<Tensor *> &rets) const override;

class Input : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  Input(const Shape &shape, const std::vector<float> &data, Device &device);
  Device *get_device() const override { return &device_; }
private:
  Shape shape_;
  std::vector<float> data_;
  Device &device_;
};

class Parameter : public Operator {
  PRIMITIV_DECL_DEFAULTS(0, 1, true);
public:
  explicit Parameter(primitiv::Parameter &param) : param_(param) {}
  Device *get_device() const override { return &param_.device(); }
  std::vector<const Tensor *> get_inner_values() const override;
private:
  primitiv::Parameter &param_;
};

class Copy : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit Copy(Device &device) : device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  Device &device_;
};

class Constant : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  Constant(const Shape &shape, float k, Device &device)
    : shape_(shape), k_(k), device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  Shape shape_;
  float k_;
  Device &device_;
};

class Identity : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  Identity(std::uint32_t size, Device &device) : size_(size), device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  std::uint32_t size_;
  Device &device_;
};

class RandomBernoulli : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  RandomBernoulli(const Shape &shape, float p, Device &device)
    : shape_(shape), p_(p), device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  Shape shape_;
  float p_;
  Device &device_;
};

class RandomUniform : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  RandomUniform(const Shape &shape, float lower, float upper, Device &device)
    : shape_(shape), lower_(lower), upper_(upper), device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  Shape shape_;
  float lower_;
  float upper_;
  Device &device_;
};

class RandomNormal : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  RandomNormal(const Shape &shape, float mean, float sd, Device &device)
    : shape_(shape), mean_(mean), sd_(sd), device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  Shape shape_;
  float mean_;
  float sd_;
  Device &device_;
};

class RandomLogNormal : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(0, 1);
public:
  RandomLogNormal(const Shape &shape, float mu, float beta, Device &device)
    : shape_(shape), mu_(mu), beta_(beta), device_(device) {}
  Device *get_device() const override { return &device_; }
private:
  Shape shape_;
  float mu_;
  float beta_;
  Device &device_;
};

class Pick : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  Pick(const std::vector<std::uint32_t> &ids, std::uint32_t dim)
    : ids_(ids), dim_(dim) {}
private:
  std::vector<std::uint32_t> ids_;
  std::uint32_t dim_;
};

class Slice : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  Slice(std::uint32_t dim, std::uint32_t lower, std::uint32_t upper)
    : dim_(dim), lower_(lower), upper_(upper) {}
private:
  std::uint32_t dim_;
  std::uint32_t lower_;
  std::uint32_t upper_;
};

class Split : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, n_);
public:
  Split(std::uint32_t dim, std::uint32_t n) : dim_(dim), n_(n) {}
private:
  std::uint32_t dim_;
  std::uint32_t n_;
};

class Concat : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(Operator::NONZERO, 1);
public:
  explicit Concat(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class Reshape : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit Reshape(const Shape &shape) : shape_(shape) {}
private:
  Shape shape_;
};

class Flip : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit Flip(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class Max : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit Max(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class Min : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit Min(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class Sum : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit Sum(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class LogSumExp : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit LogSumExp(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class Broadcast : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  Broadcast(std::uint32_t dim, std::uint32_t size) : dim_(dim), size_(size) {}
private:
  std::uint32_t dim_;
  std::uint32_t size_;
};

class SoftmaxCrossEntropy : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit SoftmaxCrossEntropy(std::uint32_t dim) : dim_(dim) {}
private:
  std::uint32_t dim_;
};

class SparseSoftmaxCrossEntropy : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit SparseSoftmaxCrossEntropy(
      const std::vector<std::uint32_t> ids,
      std::uint32_t dim) : ids_(ids), dim_(dim) {}
private:
  std::vector<std::uint32_t> ids_;
  std::uint32_t dim_;
  mutable Tensor log_softmax_x_;  // Only used when PRIMITIV_USE_CACHE=ON
};

// Unary operator with no parameter.
#define PRIMITIV_DECL_UNARY(name_) \
  class name_ : public Operator { \
    PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1); \
  }

// Unary operator with a constant.
#define PRIMITIV_DECL_UNARY_K(name_, type) \
  class name_ : public Operator { \
    PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1); \
  public: \
    explicit name_(type k) : k_(k) {} \
  private: \
    type k_; \
  }

// Binary operator with no parameter.
#define PRIMITIV_DECL_BINARY(name_) \
  class name_ : public Operator { \
    PRIMITIV_DECL_DEFAULTS_AND_FORWARD(2, 1); \
  }

PRIMITIV_DECL_UNARY(StopGradient);
PRIMITIV_DECL_UNARY(Flatten);

PRIMITIV_DECL_UNARY(Positive);
PRIMITIV_DECL_UNARY(Negative);

PRIMITIV_DECL_UNARY_K(AddConst, float);
PRIMITIV_DECL_UNARY_K(SubtractConstR, float);
PRIMITIV_DECL_UNARY_K(SubtractConstL, float);
PRIMITIV_DECL_UNARY_K(MultiplyConst, float);
PRIMITIV_DECL_UNARY_K(DivideConstR, float);
PRIMITIV_DECL_UNARY_K(DivideConstL, float);
PRIMITIV_DECL_UNARY_K(PowConstR, float);
PRIMITIV_DECL_UNARY_K(PowConstL, float);
PRIMITIV_DECL_UNARY_K(PReLU, float);
PRIMITIV_DECL_UNARY_K(ELU, float);

PRIMITIV_DECL_UNARY_K(PowN, std::int32_t);

PRIMITIV_DECL_BINARY(AddScalar);
PRIMITIV_DECL_BINARY(SubtractScalarR);
PRIMITIV_DECL_BINARY(SubtractScalarL);
PRIMITIV_DECL_BINARY(MultiplyScalar);
PRIMITIV_DECL_BINARY(DivideScalarR);
PRIMITIV_DECL_BINARY(DivideScalarL);
PRIMITIV_DECL_BINARY(PowScalarR);
PRIMITIV_DECL_BINARY(PowScalarL);

PRIMITIV_DECL_BINARY(Add);
PRIMITIV_DECL_BINARY(Subtract);
PRIMITIV_DECL_BINARY(Multiply);
PRIMITIV_DECL_BINARY(Divide);
PRIMITIV_DECL_BINARY(Pow);

PRIMITIV_DECL_UNARY(Transpose);

class PermuteDims : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit PermuteDims(const std::vector<std::uint32_t> &perm) : perm_(perm) {}
private:
  std::vector<std::uint32_t> perm_;
};

PRIMITIV_DECL_BINARY(MatrixMultiply);

PRIMITIV_DECL_UNARY(Abs);
PRIMITIV_DECL_UNARY(Sqrt);
PRIMITIV_DECL_UNARY(Exp);
PRIMITIV_DECL_UNARY(Log);
PRIMITIV_DECL_UNARY(Tanh);
PRIMITIV_DECL_UNARY(Sigmoid);
PRIMITIV_DECL_UNARY(Softplus);
PRIMITIV_DECL_UNARY(Sin);
PRIMITIV_DECL_UNARY(Cos);
PRIMITIV_DECL_UNARY(Tan);
PRIMITIV_DECL_UNARY(ReLU);
PRIMITIV_DECL_UNARY(LReLU);

class BatchPick : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  explicit BatchPick(const std::vector<std::uint32_t> &ids) : ids_(ids) {}
private:
  std::vector<std::uint32_t> ids_;
};

class BatchSlice : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  BatchSlice(std::uint32_t lower, std::uint32_t upper)
    : lower_(lower), upper_(upper) {}
private:
  std::uint32_t lower_;
  std::uint32_t upper_;
};

class BatchSplit : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, n_);
public:
  explicit BatchSplit(std::uint32_t n) : n_(n) {}
private:
  std::uint32_t n_;
};

class BatchConcat : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(Operator::NONZERO, 1);
};

PRIMITIV_DECL_UNARY(BatchSum);

class Convolution2D : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(2, 1);
public:
  Convolution2D(
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1)
  : padding0_(padding0), padding1_(padding1)
  , stride0_(stride0), stride1_(stride1)
  , dilation0_(dilation0), dilation1_(dilation1) {}
private:
  std::uint32_t padding0_, padding1_;
  std::uint32_t stride0_, stride1_;
  std::uint32_t dilation0_, dilation1_;
};

class MaxPooling2D : public Operator {
  PRIMITIV_DECL_DEFAULTS_AND_FORWARD(1, 1);
public:
  MaxPooling2D(
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1)
  : window0_(window0), window1_(window1)
  , padding0_(padding0), padding1_(padding1)
  , stride0_(stride0), stride1_(stride1) {}
private:
  std::uint32_t window0_, window1_;
  std::uint32_t padding0_, padding1_;
  std::uint32_t stride0_, stride1_;
};

#undef PRIMITIV_DECL_UNARY
#undef PRIMITIV_DECL_UNARY_K
#undef PRIMITIV_DECL_BINARY

#undef PRIMITIV_DECL_DEFAULTS_AND_FORWARD
#undef PRIMITIV_DECL_DEFAULTS

}  // namespace operators
}  // namespace primitiv

#endif  // PRIMITIV_CORE_OPERATOR_IMPL_H_
