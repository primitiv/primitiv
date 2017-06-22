#ifndef PRIMITIV_PARAMETER_H_
#define PRIMITIV_PARAMETER_H_

#include <string>
#include <vector>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>

namespace primitiv {

class Initializer;

/**
 * Class to manage a trainable tensor parameter.
 */
class Parameter {
  Parameter(const Parameter &) = delete;
  Parameter &operator=(const Parameter &) = delete;

public:
  Parameter(Parameter &&) = default;
  Parameter &operator=(Parameter &&) = default;
  
  /**
   * Creates an invalid parameter object.
   */
  Parameter() : name_(), shape_(), device_(nullptr), value_(), grad_() {}

  /**
   * Creates a new Parameter object.
   * @param name Name of the parameter.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param device The device object to manage internal memory.
   */
  Parameter(const std::string &name, const Shape &shape, Device *device);

  /**
   * Creates a new Parameter object.
   * @param name Name of the parameter.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param device The device object to manage internal memory.
   * @param value List of initial values. Order of elements should be of
   *              `Tensor::set_values()`.
   */
  Parameter(
      const std::string &name, const Shape &shape, Device *device,
      const std::vector<float> &value);

  /**
   * Creates a new Parameter object.
   * @param name Name of the parameter.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param device The device object to manage internal memory.
   * @param init An Initializer object.
   */
  Parameter(
      const std::string &name, const Shape &shape, Device *device,
      const Initializer &init);

  /**
   * Set all values.
   * @param value List of new parameter values. Order of the values should be
   *              of `Tensor::set_values()`.
   */
  void reset_value(const std::vector<float> &value);

  /**
   * Set all values using a specific initialization criteria.
   * @param init An Initializer object.
   */
  void reset_value(const Initializer &init);

  /**
   * Set all gradients to 0.
   */
  void reset_gradient();

  /**
   * Updates the value of the parameter.
   * @param diff A tensor representing the difference of each element.
   * @remarks This method performs: `value \gets value + diff`.
   */
  void add_value(const Tensor &diff);

  /**
   * Updates the gradient of the parameter.
   * @param diff A tensor representing the difference of each element.
   * @remarks This method performs: `grad \gets grad + diff`.
   */
  void add_gradient(const Tensor &diff);

  /**
   * Returns the name of the parameter.
   * @return Name of the parameter.
   */
  const std::string &name() const { return name_; }

  /**
   * Returns the shape of the parameter.
   * @return Shape object.
   */
  const Shape &shape() const { return shape_; }

  /**
   * Returns the Device object to manage the internal memory.
   * @return Pointer of the Device object.
   */
  Device *device() const { return device_; }

  /**
   * Returns the values of the parameter.
   * @return A tensor representing the parameter tensor.
   */
  const Tensor &value() const { return value_; }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  const Tensor &gradient() const { return grad_; }

private:
  /**
   * Check shape of the parameter.
   */
  void check_shape();

  std::string name_;
  Shape shape_;
  Device *device_;
  Tensor value_;
  Tensor grad_;
};

}  // namespace primitiv

#endif  // PRIMITIV_PARAMETER_H_
