#include <config.h>

#include <vector>
#include <primitiv/error.h>
#include <primitiv/shape_ops.h>

namespace primitiv {
namespace shape_ops {

Shape slice(const Shape &x, unsigned dim, unsigned lower, unsigned upper) {
  if (lower >= upper || upper > x.dim(dim)) {
    THROW_ERROR(
        "Invalid slice operation. shape: " << x.to_string()
        << ", dim: " << dim << ", lower: " << lower << ", upper: " << upper);
  }

  if (dim >= x.dims().size()) return x;

  std::vector<unsigned> dims = x.dims();
  dims[dim] = upper - lower;
  return Shape(dims, x.batch_size());
}

}  // namespace shape_ops
}  // namespace primitiv
