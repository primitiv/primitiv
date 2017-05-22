#ifndef PRIMITIV_INITIALIZER_H_
#define PRIMITIV_INITIALIZER_H_

namespace primitiv {

class Tensor;

/**
 * Abstract class to provide parameter initialization algorithms.
 */
class Initializer {
  Initializer(const Initializer &) = delete;
  Initializer(Initializer &&) = delete;
  Initializer &operator=(const Initializer &) = delete;
  Initializer &operator=(Initializer &&) = delete;

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
