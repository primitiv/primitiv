#include <config.h>

#include <primitiv/error.h>
#include <primitiv/shape_ops.h>

namespace primitiv {
namespace shape_ops {

Shape slice(const Shape &x, unsigned dim, unsigned lower, unsigned upper) {
  if (lower >= upper || upper > x[dim]) {
    THROW_ERROR(
        "Invalid slice operation. shape: " << x.to_string()
        << ", dim: " << dim << ", lower: " << lower << ", upper: " << upper);
  }

  return dim >= x.depth() ? x : x.resize_dim(dim, upper - lower);
}

Shape concat(const std::vector<const Shape *> &xs) {
  THROW_ERROR("not implemented");
}

}  // namespace shape_ops
}  // namespace primitiv
