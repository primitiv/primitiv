#ifndef PRIMITIV_FUNCTION_H_
#define PRIMITIV_FUNCTION_H_

#include <string>
#include <vector>
#include <primitiv/mixins.h>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>

namespace primitiv {

class Device;

/**
 * Interface of the function on the computation graph.
 */
class Function : mixins::Nonmovable<Function> {
public:
  Function() = default;
  virtual ~Function() = default;

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
   * Returns the pre-calculated return value of the function if they have it.
   * @return A pointer of the Tensor which represents the return value, or
   *         nullptr if the function does not have such data.
   * @remarks If this function returns nullptr, the return values of the
   *          function could be obtained through forward().
   */
  virtual const Tensor *get_inner_value() const { return nullptr; }

  /**
   * Calculates the forward path.
   * @param args argument tensors.
   * @return Resulting tensors.
   * @remarks This function is not const-qualified because some function
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
   * Returns the name of the function.
   * @return Name of the function.
   */
  virtual std::string name() const = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_H_
