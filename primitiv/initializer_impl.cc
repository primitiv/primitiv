#include <config.h>

#include <primitiv/initializer_impl.h>
#include <primitiv/tensor.h>

namespace primitiv {
namespace initializers {

void Constant::apply(Tensor &x) const {
  x.reset(k_);
}

}  // namespace initializers
}  // namespace primitiv
