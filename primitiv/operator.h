#ifndef PRIMITIV_OPERATOR_H_
#define PRIMITIV_OPERATOR_H_

#include <string>
#include <vector>
#include <primitiv/mixins.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>

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
   * Calculates only the resulting shape.
   * @pasram args Shapes of argument values.
   * @return Shape of the resulting value.
   */
  virtual Shape forward_shape(
      const std::vector<const Shape *> &args) const = 0;

  /**
   * Returns the device object if the class holds it.
   * @return A pointer of the Device object if the class holds it, or nullptr
   *         otherwise.
   */
  virtual Device *get_device() const { return nullptr; }

  /**
   * Returns the pre-calculated return value of the Operator if they have it.
   * @return A pointer of the Tensor which represents the return value, or
   *         nullptr if the Operator does not have such data.
   * @remarks If this function returns nullptr, the return values of the
   *          Operator could be obtained through forward().
   */
  virtual const Tensor *get_inner_value() const { return nullptr; }

  /**
   * Calculates the forward path.
   * @param args argument tensors.
   * @return Resulting tensors.
   * @remarks This function is not const-qualified because some Operator
   *          implementations may hold the cache of intermediate results.
   */
  virtual Tensor forward(const std::vector<const Tensor *> &args) = 0;

  /**
   * Calculates the backward path.
   * @param cur_value The value of the current node.
   * @param cur_grad The gradient of the current node.
   * @param arg_values Values of the argument nodes.
   * @param arg_grads Gradients of the argument nodes. These values are updated
   *                  by this method.
   */
  virtual void backward(
      const Tensor &cur_value,
      const Tensor &cur_grad,
      const std::vector<const Tensor *> &arg_values,
      const std::vector<Tensor *> &arg_grads) const = 0;

  /**
   * Returns the name of the Operator.
   * @return Name of the Operator.
   */
  virtual std::string name() const = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_OPERATOR_H_
