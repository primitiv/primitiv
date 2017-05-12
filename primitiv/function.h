#ifndef PRIMITIV_FUNCTION_H_
#define PRIMITIV_FUNCTION_H_

#include <vector>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Interface of the function on the computation graph.
 */
class Function {
  public:
    /**
     * Calculates only the resulting shape.
     * @pasram args Shapes of argument values.
     * @return Shape of the resulting value.
     */
    virtual Shape forward_shape(const std::vector<const Shape *> &args) = 0;

    /**
     * Calculates forwarding path.
     * @param args argument tensors.
     * @return Resulting tensors.
     */
    virtual Tensor forward(const std::vector<const Tensor *> &args) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_H_
