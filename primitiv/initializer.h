#ifndef PRIMITIV_INITIALIZER_H_
#define PRIMITIV_INITIALIZER_H_

#include <primitiv/mixins.h>

namespace primitiv {

class Tensor;

/**
 * Abstract class to provide parameter initialization algorithms.
 */
class Initializer : mixins::Nonmovable<Initializer> {
public:
  Initializer() = default;
  virtual ~Initializer() = default;

  /**
   * Provides an initialized tensor.
   * @param x Tensor object to be initialized.
   */
  virtual void apply(Tensor &x) const = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_INITIALIZER_H_
