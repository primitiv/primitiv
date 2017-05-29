#ifndef PRIMITIV_INITIALIZER_IMPL_H_
#define PRIMITIV_INITIALIZER_IMPL_H_

#include <primitiv/initializer.h>

namespace primitiv {
namespace initializers {

/**
 * Initializer class to generate a same-value tensor.
 */
class Constant : public Initializer {
  Constant() = delete;
  Constant(const Constant &) = delete;
  Constant(Constant &&) = delete;
  Constant &operator=(const Constant &) = delete;
  Constant &operator=(Constant &&) = delete;

public:
  /**
   * Crates a new initializer object.
   * @param k Constant to provide.
   */
  Constant(const float k) : k_(k) {}
  ~Constant() override = default;

  void apply(Tensor &x) const override;

private:
  float k_;
};

}  // namespace initializers
}  // namespace primitiv

#endif  // PRIMITIV_INITIALIZER_IMPL_H_
