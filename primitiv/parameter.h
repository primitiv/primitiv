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
   * @param value List of initial values. Order of elements should be of
   *              `Tensor::set_values()`.
   * @param device The device object to manage internal memory.
   */
  Parameter(
      const std::string &name, const Shape &shape,
      const std::vector<float> &value,
      Device *device);

  /**
   * Creates a new Parameter object.
   * @param name Name of the parameter.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param init An Initializer object.
   * @param device The device object to manage internal memory.
   */
  Parameter(
      const std::string &name, const Shape &shape,
      const Initializer &init,
      Device *device);

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
   * Returns the values of the parameter.
   * @return A tensor representing the parameter tensor.
   */
  Tensor &value() { return value_; }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  const Tensor &gradient() const { return grad_; }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  Tensor &gradient() { return grad_; }

  /**
   * Loads parameters and returns a new Parameter object.
   * @param path File path to load parameters.
   * @param device Device object to manage internal memories.
   * @return A new Parameter object.
   */
  static Parameter load(const std::string &path, Device *device);

  /**
   * Saves current parameters into specified file with YAML format.
   * @param path File path to write parameters.
   */
  void save(const std::string &path) const;

private:
  /**
   * Makes a Parameter object directly from its values.
   * @param name Name of the parameter.
   * @param value Value of the parameter.
   */
  Parameter(const std::string &name, Tensor &&value);

  /**
   * Checks the shape of the parameter.
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
