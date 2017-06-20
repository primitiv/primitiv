#include <config.h>

#include <algorithm>
#include <primitiv/error.h>
#include <primitiv/shape_ops.h>

namespace primitiv {
namespace shape_ops {

Shape elementwise(const Shape &a, const Shape &b) {
  if (!a.has_same_dims(b) || !a.has_compatible_batch(b)) {
    THROW_ERROR(
        "Shape mismatched. a: " << a.to_string() << " != b: " << b.to_string());
  }
  return a.resize_batch(std::max(a.batch_size(), b.batch_size()));
}

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

Shape broadcast(const Shape &x, unsigned dim, unsigned size) {
  if (x[dim] != 1 || size == 0) {
    THROW_ERROR(
        "Invalid broadcasting. x: "
        << x.to_string() << ", dim: " << dim << ", size: " << size);
  }
  return x.resize_dim(dim, size);
}

Shape pick(const Shape &x, unsigned dim, const std::vector<unsigned> &ids) {
  const unsigned n = x[dim];
  const unsigned bx = x.batch_size();
  const unsigned bi = ids.size();
  if (bi == 0 || (bx != bi && bx > 1 && bi > 1)) {
    THROW_ERROR(
        "Invalid IDs to pick. shape: " << x.to_string()
        << ", ids.size(): " << ids.size());
  }
  for (unsigned i = 0; i < bi; ++i) {
    if (ids[i] >= n) {
      THROW_ERROR(
          "Invalid IDs to pick. shape: " << x.to_string()
          << ", ids[" << i << "]: " << ids[i]);
    }
  }

  Shape ret = x.resize_dim(dim, 1);
  ret.update_batch(std::max(bx, bi));
  return ret;
}

Shape transpose(const Shape &x) {
  if (x.depth() > 2) {
    THROW_ERROR("Invalid shape to transpose: " << x.to_string());
  }
  return Shape({x[1], x[0]}, x.batch_size());
}

Shape dot(const Shape &l, const Shape &r) {
  if (l.depth() > 2 || r.depth() > 2 || l[1] != r[0] ||
      !l.has_compatible_batch(r)) {
    THROW_ERROR(
        "Invalid shapes to calculate the dot product: "
        << l.to_string() << ", " << r.to_string());
  }
  return Shape({l[0], r[1]}, std::max(l.batch_size(), r.batch_size()));
}

}  // namespace shape_ops
}  // namespace primitiv
