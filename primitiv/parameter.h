#ifndef PRIMITIV_PARAMETER_H_
#define PRIMITIV_PARAMETER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <primitiv/error.h>
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
  Parameter(Parameter &&src)
    : name_(std::move(src.name_))
    , shape_(std::move(src.shape_))
    , device_(src.device_)
    , value_(std::move(src.value_))
    , grad_(std::move(src.grad_))
    , stats_(std::move(src.stats_)) {
      src.device_ = nullptr;
    }

  Parameter &operator=(Parameter &&src) {
    if (&src != this) {
      name_ = std::move(src.name_);
      shape_ = std::move(src.shape_);
      device_ = src.device_;
      value_ = std::move(src.value_);
      grad_ = std::move(src.grad_);
      stats_ = std::move(src.stats_);
      src.device_ = nullptr;
    }
    return *this;
  }

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
  Parameter(const std::string &name, const Shape &shape, Device &device);

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
      Device &device);

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
      Device &device);

  /**
   * Returns whether the parameter is valid or not.
   * @return true or false w.r.t. the parameter is valid or not.
   */
  bool valid() const { return !!device_; }

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
   * Adds a new optional statistics tensor.
   * @param name Name of the statistics.
   * @param shape Shape of the tensor.
   */
  void add_stats(const std::string &name, const Shape &shape);

  /**
   * Checks whether the statistics with name `name` exists or not.
   * @param name Name of the statistics.
   * @return true if the entry exists, false otherwise.
   */
  bool has_stats(const std::string &name) const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return stats_.find(name) != stats_.end();
  }

  /**
   * Returns the name of the parameter.
   * @return Name of the parameter.
   */
  const std::string &name() const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return name_;
  }

  /**
   * Returns the shape of the parameter.
   * @return Shape object.
   */
  const Shape &shape() const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return shape_;
  }

  /**
   * Returns the Device object to manage the internal memory.
   * @return Pointer of the Device object.
   */
  Device &device() const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return *device_;
  }

  /**
   * Returns the values of the parameter.
   * @return A tensor representing the parameter tensor.
   */
  const Tensor &value() const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return value_;
  }

  /**
   * Returns the values of the parameter.
   * @return A tensor representing the parameter tensor.
   */
  Tensor &value() {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return value_;
  }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  const Tensor &gradient() const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return grad_;
  }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  Tensor &gradient() {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return grad_; }

  /**
   * Returns the current opotional statistics tensor specified by given name.
   * @param name Name of the statistics.
   * @return A tensor.
   */
  const Tensor &stats(const std::string &name) const {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return stats_.at(name);
  }

  /**
   * Returns the current opotional statistics tensor specified by given name.
   * @param name Name of the statistics.
   * @return A tensor.
   */
  Tensor &stats(const std::string &name) {
    if (!valid()) THROW_ERROR("Invalid parameter.");
    return stats_.at(name);
  }

  /**
   * Loads parameters and returns a new Parameter object.
   * @param path File path to load parameters.
   * @param device Device object to manage internal memories.
   * @return A new Parameter object.
   */
  static Parameter load(const std::string &path, Device &device);

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
   * @param stats Map of optional statistics.
   */
  Parameter(
      std::string &&name, Tensor &&value,
      std::unordered_map<std::string, Tensor> &&stats);

  /**
   * Checks the shape of the parameter.
   */
  void check_shape();

  std::string name_;
  Shape shape_;
  Device *device_;
  Tensor value_;
  Tensor grad_;
  std::unordered_map<std::string, Tensor> stats_;
};

}  // namespace primitiv

#endif  // PRIMITIV_PARAMETER_H_
