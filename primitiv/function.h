#ifndef PRIMITIV_FUNCTION_H_
#define PRIMITIV_FUNCTION_H_

#include <string>
#include <vector>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Interface of the function on the computation graph.
 */
class Function {
public:
  virtual ~Function() = default;

  /**
   * Calculates only the resulting shape.
   * @pasram args Shapes of argument values.
   * @return Shape of the resulting value.
   */
  virtual Shape forward_shape(
      const std::vector<const Shape *> &args) const = 0;

  /**
   * Calculates the forward path.
   * @param args argument tensors.
   * @return Resulting tensors.
   */
  virtual Tensor forward(const std::vector<const Tensor *> &args) const = 0;

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
