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

Shape concat(const std::vector<const Shape *> &xs, unsigned dim) {
  if (xs.empty()) {
    THROW_ERROR("No tensors to be concatenated.");
  }

  Shape s0 = *xs[0];
  unsigned sum = s0[dim];

  for (unsigned i = 1; i < xs.size(); ++i) {
    const Shape &s = *xs[i];
    if (!s0.has_same_loo_dims(s, dim) || !s0.has_compatible_batch(s)) {
      std::string dims_str = xs[0]->to_string();
      for (unsigned i = 1; i < xs.size(); ++i) {
        dims_str += ", " + xs[i]->to_string();
      }
      THROW_ERROR("Invalid shapes to concatenate: " << dims_str);
    }
    if (s0.batch_size() == 1) s0.update_batch(s.batch_size());
    sum += s[dim];
  }

  s0.update_dim(dim, sum);
  return s0;
}

}  // namespace shape_ops
}  // namespace primitiv
