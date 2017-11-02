#ifndef PRIMITIV_INITIALIZER_IMPL_H_
#define PRIMITIV_INITIALIZER_IMPL_H_

#include <primitiv/initializer.h>

namespace primitiv {
namespace initializers {

/**
 * Initializer to generate a same-value tensor.
 */
class Constant : public Initializer {
  Constant() = delete;

public:
  /**
   * Crates a new initializer object.
   * @param k Constant to provide.
   */
  explicit Constant(float k) : k_(k) {}

  void apply(Tensor &x) const override;

private:
  float k_;
};

/**
 * Initializer using a parameterized uniform distribution (lower, upper].
 */
class Uniform : public Initializer {
public:
  Uniform(float lower, float upper) : lower_(lower), upper_(upper) {}

  void apply(Tensor &x) const override;

private:
  float lower_;
  float upper_;
};

/**
 * Initializer using a parameterized normal distribution N(mean, sd).
 */
class Normal : public Initializer {
public:
  Normal(float mean, float sd) : mean_(mean), sd_(sd) {}

  void apply(Tensor &x) const override;

private:
  float mean_;
  float sd_;
};

/**
 * Identity matrix initializer.
 */
class Identity : public Initializer {
public:
  Identity() {}

  void apply(Tensor &x) const override;
};

/**
 * The Xavier matrix initialization with the uniform distribution.
 */
class XavierUniform : public Initializer {
public:
  XavierUniform(float scale = 1.0f) : scale_(scale) {}

  void apply(Tensor &x) const override;

private:
  float scale_;
};

/**
 * The Xavier matrix initialization with the normal distribution.
 */
class XavierNormal : public Initializer {
public:
  XavierNormal(float scale = 1.0f) : scale_(scale) {}

  void apply(Tensor &x) const override;

private:
  float scale_;
};

}  // namespace initializers
}  // namespace primitiv

#endif  // PRIMITIV_INITIALIZER_IMPL_H_
