#ifndef PRIMITIV_CORE_OPERATOR_H_
#define PRIMITIV_CORE_OPERATOR_H_

#include <string>
#include <vector>

#include <primitiv/core/mixins/nonmovable.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>

namespace primitiv {

class Device;

/**
 * Interface of the operator on the computation graph.
 */
class Operator : mixins::Nonmovable<Operator> {
public:
  Operator() = default;
  virtual ~Operator() = default;

  /**
   * Returns the name of the Operator.
   * @return Name of the Operator.
   */
  virtual std::string name() const = 0;

  static constexpr std::uint32_t ANY = 0xffffffff;
  static constexpr std::uint32_t NONZERO = 0xfffffffe;

  /**
   * Retrieves the number of required arguments.
   * @return Number of required arguments, or following special values:
   *           Operator::ANY: Arbitrary number of arguments, including 0
   *           Operator::NONZERO: Arbitrary number of arguments, except 0
   */
  virtual std::uint32_t num_arguments() const = 0;

  /**
   * Retrieves the number of return values.
   * @return Number of return values.
   */
  virtual std::uint32_t num_returns() const = 0;

  /**
   * Returns whether the operator have inner values or not.
   * @return `true` if the operator have inner values,
   *         `false` otherwise.
   */
  virtual bool has_inner_values() const = 0;

  /**
   * Returns the device object if the class holds it.
   * @return A pointer of the Device object if the class holds it, or nullptr
   *         otherwise.
   */
  virtual Device *get_device() const { return nullptr; }

  /**
   * Calculates only the resulting shape.
   * @param args Shapes of argument values.
   * @param rets Shapes of return values.
   * @remarks `args` and `rets` should have the same number of pointers with the
   *          value returned from `num_arguments()` and `num_returns()`.
   */
  virtual void forward_shape(
      const std::vector<const Shape *> &args,
      const std::vector<Shape *> &rets) const = 0;

  /**
   * Returns the pre-calculated return value of the operator.
   * @return A list of pointers of the Tensor which represents the return
   *         values.
   * @throw primitiv::Error The operator does not have such values.
   */
  virtual std::vector<const Tensor *> get_inner_values() const {
    PRIMITIV_THROW_ERROR(
        "Operator `" << name()
        << "` does not have inner values. Use `forward()` instead.");
  }

  /**
   * Calculates the forward operation.
   * @param args Argument tensors.
   * @param rets Resulting tensors.
   * @return Resulting tensors.
   * @remarks `args` and `rets` should have the same number of pointers with the
   *          value returned from `num_arguments()` and `num_returns()`.
   */
  virtual void forward(
      const std::vector<const Tensor *> &args,
      const std::vector<Tensor *> &rets) const {
    static_cast<void>(args);
    static_cast<void>(rets);
    PRIMITIV_THROW_ERROR(
        "Operator `" << name()
        << "`has inner values. Use `get_inner_values()` instead.");
  }

  /**
   * Calculates the backward operation.
   * @param args_v Tensors of argument values.
   * @param rets_v Tensors of resulting values.
   * @param rets_g Tensors of gradients of results.
   * @param args_g Tensors of gradients of arguments.
   * @remarks `args_v/g` and `rets_v/g` should have the same number of pointers
   *          with the value returned from `num_arguments()` and
   *          `num_returns()`.
   */
  virtual void backward(
      const std::vector<const Tensor *> &args_v,
      const std::vector<const Tensor *> &rets_v,
      const std::vector<const Tensor *> &rets_g,
      const std::vector<Tensor *> &args_g) const = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_OPERATOR_H_
