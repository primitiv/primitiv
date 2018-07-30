#ifndef PRIMITIV_CORE_PARAMETER_H_
#define PRIMITIV_CORE_PARAMETER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/nonmovable.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>
#include <primitiv/msgpack/reader.h>
#include <primitiv/msgpack/writer.h>

namespace primitiv {

class Device;
class Initializer;

/**
 * Class to manage a trainable tensor parameter.
 */
class Parameter : mixins::Nonmovable<Parameter> {
  friend class Model;

private:
  /**
   * Loads parameters from msgpack::Reader w/o checking the header.
   * @param reader msgpack::Reader object.
   * @param with_stats Whether or not to load all additional statistics.
   * @param device Device object to manage the parameter.
   */
  void load_inner(msgpack::Reader &reader, bool with_stats, Device &device);

  /**
   * Saves parameters to msgpack::Writer.
   * @param writer msgpack::Writer object.
   * @param with_stats Whether or not to save all additional statistics.
   */
  void save_inner(msgpack::Writer &writer, bool with_stats) const;

public:
  /**
   * Creates an invalid parameter object.
   */
  Parameter() : shape_(), device_(nullptr), value_(), grad_() {}

  /**
   * Creates a new Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param value List of initial values. Order of elements should be the
   *              column-major (Fortran) order.
   * @param device The device object to manage internal memory.
   */
  Parameter(
      const Shape &shape, const std::vector<float> &value, Device *device);

  /**
   * Creates a new Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param value List of initial values. Order of elements should be the
   *              column-major (Fortran) order.
   * @param device The device object to manage internal memory.
   */
  Parameter(
      const Shape &shape, const std::vector<float> &value, Device &device)
    : Parameter(shape, value, &device) {}

  /**
   * Creates a new Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param value List of initial values. Order of elements should be the
   *              column-major (Fortran) order.
   */
  Parameter(const Shape &shape, const std::vector<float> &value)
    : Parameter(shape, value, nullptr) {}

  /**
   * Creates a new Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param initializer An Initializer object.
   * @param device The device object to manage internal memory.
   */
  Parameter(
      const Shape &shape, const Initializer &initializer, Device *device);

  /**
   * Creates a new Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param initializer An Initializer object.
   * @param device The device object to manage internal memory.
   */
  Parameter(
      const Shape &shape, const Initializer &initializer, Device &device)
    : Parameter(shape, initializer, &device) {}

  /**
   * Creates a new Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param initializer An Initializer object.
   */
  Parameter(const Shape &shape, const Initializer &initializer)
    : Parameter(shape, initializer, nullptr) {}

  /**
   * Initializes the Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param value List of initial values. Order of elements should be the
   *              column-major (Fortran) order.
   * @param device The device object to manage internal memory.
   */
  void init(
      const Shape &shape, const std::vector<float> &value, Device *device);

  /**
   * Initializes the Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param value List of initial values. Order of elements should be the
   *              column-major (Fortran) order.
   * @param device The device object to manage internal memory.
   */
  void init(
      const Shape &shape, const std::vector<float> &value, Device &device) {
    init(shape, value, &device);
  }

  /**
   * Initializes the Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param value List of initial values. Order of elements should be the
   *              column-major (Fortran) order.
   */
  void init(const Shape &shape, const std::vector<float> &value) {
    init(shape, value, nullptr);
  }

  /**
   * Initializes the Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param initializer An Initializer object.
   * @param device The device object to manage internal memory.
   */
  void init(
      const Shape &shape, const Initializer &initializer, Device *device);

  /**
   * Initializes the Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param initializer An Initializer object.
   * @param device The device object to manage internal memory.
   */
  void init(
      const Shape &shape, const Initializer &initializer, Device &device) {
    init(shape, initializer, &device);
  }

  /**
   * Initializes the Parameter object.
   * @param shape The shape of the parameter. The batch size should be 1.
   * @param initializer An Initializer object.
   */
  void init(
      const Shape &shape, const Initializer &initializer) {
    init(shape, initializer, nullptr);
  }

  /**
   * Loads parameters from specified file.
   * @param path File path to load parameters.
   * @param with_stats Whether or not to load all additional statistics as well
   *                   as parameter values if the file has them.
   * @param device The device object to manage internal memory.
   */
  void load(const std::string &path, bool with_stats, Device *device);

  /**
   * Loads parameters from specified file.
   * @param path File path to load parameters.
   * @param with_stats Whether or not to load all additional statistics as well
   *                   as parameter values if the file has them.
   * @param device The device object to manage internal memory.
   */
  void load(const std::string &path, bool with_stats, Device &device) {
    load(path, with_stats, &device);
  }

  /**
   * Loads parameters from specified file.
   * @param path File path to load parameters.
   * @param with_stats Whether or not to load all additional statistics as well
   *                   as parameter values if the file has them.
   */
  void load(const std::string &path, bool with_stats) {
    load(path, with_stats, nullptr);
  }

  /**
   * Loads parameters from specified file.
   * @param path File path to load parameters.
   */
  void load(const std::string &path) {
    load(path, true, nullptr);
  }

  /**
   * Saves current parameters into specified file.
   * @param path File path to save parameters.
   * @param with_stats Whether or not to save all additional statistics as well
   *                   as parameter values if the parameter object has them.
   */
  void save(const std::string &path, bool with_stats) const;

  /**
   * Saves current parameters into specified file.
   * @param path File path to save parameters.
   */
  void save(const std::string &path) const {
    save(path, true);
  }

  /**
   * Returns whether the parameter is valid or not.
   * @return true or false w.r.t. the parameter is valid or not.
   */
  bool valid() const { return !!device_; }

  /**
   * Set all gradients to 0.
   */
  void reset_gradient();

  /**
   * Adds a new optional statistics tensor.
   * @param name Name of the statistics.
   * @param shape Shape of the tensor.
   * @remarks All elements in the new statistics tensor is initialized by 0.
   */
  void add_stats(const std::string &name, const Shape &shape);

  /**
   * Checks whether the statistics with name `name` exists or not.
   * @param name Name of the statistics.
   * @return true if the entry exists, false otherwise.
   */
  bool has_stats(const std::string &name) const {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return stats_.find(name) != stats_.end();
  }

  /**
   * Returns the shape of the parameter.
   * @return Shape object.
   */
  Shape shape() const {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return shape_;
  }

  /**
   * Returns the Device object to manage the internal memory.
   * @return Pointer of the Device object.
   */
  Device &device() const {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return *device_;
  }

  /**
   * Returns the values of the parameter.
   * @return A tensor representing the parameter tensor.
   */
  const Tensor &value() const {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return value_;
  }

  /**
   * Returns the values of the parameter.
   * @return A tensor representing the parameter tensor.
   */
  Tensor &value() {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return value_;
  }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  const Tensor &gradient() const {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return grad_;
  }

  /**
   * Returns the current gradient of the parameter.
   * @return A tensor representing the gradient of the value.
   */
  Tensor &gradient() {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return grad_; }

  /**
   * Returns the current opotional statistics tensor specified by given name.
   * @param name Name of the statistics.
   * @return A tensor.
   */
  const Tensor &stats(const std::string &name) const {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return stats_.at(name);
  }

  /**
   * Returns the current opotional statistics tensor specified by given name.
   * @param name Name of the statistics.
   * @return A tensor.
   */
  Tensor &stats(const std::string &name) {
    if (!valid()) PRIMITIV_THROW_ERROR("Invalid parameter.");
    return stats_.at(name);
  }

private:
  Shape shape_;
  Device *device_;
  Tensor value_;
  Tensor grad_;
  std::unordered_map<std::string, Tensor> stats_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_PARAMETER_H_
