#ifndef PRIMITIV_CONSTANT_INITIALIZER_H_
#define PRIMITIV_CONSTANT_INITIALIZER_H_

#include <primitiv/initializer.h>

namespace primitiv {

/**
 * Initializer class to generate a same-value tensor.
 */
class ConstantInitializer : public Initializer {
  ConstantInitializer() = delete;
  ConstantInitializer(const ConstantInitializer &) = delete;
  ConstantInitializer(ConstantInitializer &&) = delete;
  ConstantInitializer &operator=(const ConstantInitializer &) = delete;
  ConstantInitializer &operator=(ConstantInitializer &&) = delete;

public:
  /**
   * Crates a new ConstantInitializer object.
   * @param k Constant to provide.
   */
  ConstantInitializer(const float k) : k_(k) {}
  ~ConstantInitializer() override = default;

  Tensor generate(const Shape &shape, Device *device) const override;

private:
  float k_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CONSTANT_INITIALIZER_H_
