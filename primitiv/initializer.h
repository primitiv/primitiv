#ifndef PRIMITIV_INITIALIZER_H_
#define PRIMITIV_INITIALIZER_H_

#include <primitiv/tensor.h>

namespace primitiv {

class Device;
class Shape;

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
   * @param shape Shape of the tensor.
   * @param device Device object to manage the tensor.
   * @return Generated Tensor object.
   */
  virtual Tensor generate(const Shape &shape, Device *device) const = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_INITIALIZER_H_
