#ifndef PRIMITIV_CORE_INITIALIZER_IMPL_H_
#define PRIMITIV_CORE_INITIALIZER_IMPL_H_

#include <primitiv/core/initializer.h>

namespace primitiv {
namespace initializers {

/**
 * Initializer to generate a same-value tensor.
 */
class Constant : public Initializer {
  Constant() = delete;

public:
  /**
   * Crates a new ``Constant`` initializer.
   * @param k Initial value of all variables in the parameter.
   */
  explicit Constant(float k) : k_(k) {}

  void apply(Tensor &x) const override;

private:
  float k_;
};

/**
 * Initializer using a parameterized uniform distribution with the range
 * \f$ (L, U] \f$.
 */
class Uniform : public Initializer {
public:
  /**
   * Creates a new ``Uniform`` initializer.
   * @param lower Lower bound \f$ L \f$ of the uniform distribution.
   * @param upper Upper bound \f$ U \f$ of the uniform distribution.
   */
  Uniform(float lower, float upper) : lower_(lower), upper_(upper) {}

  void apply(Tensor &x) const override;

private:
  float lower_;
  float upper_;
};

/**
 * Initializer using a parameterized normal distribution
 * \f$ \mathcal{N}(\mu, \sigma) \f$.
 */
class Normal : public Initializer {
public:
  /**
   * Creates a new ``Normal`` initializer.
   * @param mean Mean \f$ \mu \f$ of the normal distribution.
   * @param sd Standard deviation \f$ \sigma \f$ of the normal distribution.
   */
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
  /**
   * Creates a new ``Identity`` initializer.
   */
  Identity() {}

  void apply(Tensor &x) const override;
};

/**
 * The Xavier matrix initialization with the uniform distribution.
 */
class XavierUniform : public Initializer {
public:
  /**
   * Creates a new ``XavierUniform`` initializer.
   * @param scale Additional scaling factor of the uniform distribution.
   */
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
  /**
   * Creates a new ``XavierNormal`` initializer.
   * @param scale Additional scaling factor of the normal distribution.
   */
  XavierNormal(float scale = 1.0f) : scale_(scale) {}

  void apply(Tensor &x) const override;

private:
  float scale_;
};

/**
 * The Xavier initialization with the uniform distribution for conv2d filters.
 */
class XavierUniformConv2D : public Initializer {
public:
  /**
   * Creates a new ``XavierUniformConv2D`` initializer.
   * @param scale Additional scaling factor of the uniform distribution.
   */
  XavierUniformConv2D(float scale = 1.0f) : scale_(scale) {}

  void apply(Tensor &x) const override;

private:
  float scale_;
};

/**
 * The Xavier initialization with the normal distribution for conv2d filters.
 */
class XavierNormalConv2D : public Initializer {
public:
  /**
   * Creates a new ``XavierNormalConv2D`` initializer.
   * @param scale Additional scaling factor of the normal distribution.
   */
  XavierNormalConv2D(float scale = 1.0f) : scale_(scale) {}

  void apply(Tensor &x) const override;

private:
  float scale_;
};

}  // namespace initializers
}  // namespace primitiv

#endif  // PRIMITIV_CORE_INITIALIZER_IMPL_H_
