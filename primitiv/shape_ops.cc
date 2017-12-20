#include <primitiv/config.h>

#include <algorithm>
#include <primitiv/error.h>
#include <primitiv/shape_ops.h>

namespace primitiv {
namespace shape_ops {

Shape reshape(const Shape &before, const Shape &after) {
  if (before.volume() != after.volume() ||
      (after.has_batch() && after.batch() != before.batch())) {
    THROW_ERROR(
        "Invalid shapes to reshape. before: " << before.to_string()
        << ", after: " << after.to_string());
  }
  return after.resize_batch(before.batch());
}

Shape flatten(const Shape &x) {
  return Shape({x.volume()}, x.batch());
}

Shape scalar_op(const Shape &x, const Shape &k) {
  if (!k.is_scalar() || !x.has_compatible_batch(k)) {
    THROW_ERROR(
        "Shape mismatched for the scalar operation. "
        "x: " << x.to_string() << " != k: " << k.to_string());
  }
  return x.resize_batch(std::max(x.batch(), k.batch()));
}

Shape elementwise(const Shape &a, const Shape &b) {
  if (!a.has_same_dims(b) || !a.has_compatible_batch(b)) {
    THROW_ERROR(
        "Shape mismatched for the elementwise operation. "
        "a: " << a.to_string() << " != b: " << b.to_string());
  }
  return a.resize_batch(std::max(a.batch(), b.batch()));
}

Shape slice(const Shape &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper) {
  if (lower >= upper || upper > x[dim]) {
    THROW_ERROR(
        "Invalid slice operation. shape: " << x.to_string()
        << ", dim: " << dim << ", lower: " << lower << ", upper: " << upper);
  }

  return dim >= x.depth() ? x : x.resize_dim(dim, upper - lower);
}

Shape concat(const std::vector<Shape> &xs, std::uint32_t dim) {
  std::vector<const Shape *> ptrs;
  ptrs.reserve(xs.size());
  for (const Shape &x : xs) ptrs.emplace_back(&x);
  return concat(ptrs, dim);
}

Shape concat(const std::vector<const Shape *> &xs, std::uint32_t dim) {
  if (xs.empty()) {
    THROW_ERROR("No tensors to be concatenated.");
  }

  Shape s0 = *xs[0];
  std::uint32_t sum = s0[dim];

  for (std::uint32_t i = 1; i < xs.size(); ++i) {
    const Shape &s = *xs[i];
    if (!s0.has_same_loo_dims(s, dim) || !s0.has_compatible_batch(s)) {
      std::string dims_str = xs[0]->to_string();
      for (std::uint32_t i = 1; i < xs.size(); ++i) {
        dims_str += ", " + xs[i]->to_string();
      }
      THROW_ERROR("Invalid shapes to concatenate: " << dims_str);
    }
    if (!s0.has_batch()) s0.update_batch(s.batch());
    sum += s[dim];
  }

  s0.update_dim(dim, sum);
  return s0;
}

Shape broadcast(const Shape &x, std::uint32_t dim, std::uint32_t size) {
  if (x[dim] != 1 || size == 0) {
    THROW_ERROR(
        "Invalid broadcasting. x: "
        << x.to_string() << ", dim: " << dim << ", size: " << size);
  }
  return x.resize_dim(dim, size);
}

Shape pick(const Shape &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  const std::uint32_t n = x[dim];
  const std::uint32_t bi = ids.size();
  if (bi == 0 || (x.batch() != bi && x.has_batch() && bi > 1)) {
    THROW_ERROR(
        "Invalid IDs to pick. shape: " << x.to_string()
        << ", ids.size(): " << ids.size());
  }
  for (std::uint32_t i = 0; i < bi; ++i) {
    if (ids[i] >= n) {
      THROW_ERROR(
          "Invalid IDs to pick. shape: " << x.to_string()
          << ", ids[" << i << "]: " << ids[i]);
    }
  }

  Shape ret = x.resize_dim(dim, 1);
  ret.update_batch(std::max(x.batch(), bi));
  return ret;
}

Shape transpose(const Shape &x) {
  if (!x.is_matrix()) {
    THROW_ERROR("Invalid shape to transpose: " << x.to_string());
  }
  return Shape({x[1], x[0]}, x.batch());
}

Shape matmul(const Shape &l, const Shape &r) {
  if (!l.is_matrix() || !r.is_matrix() || l[1] != r[0] ||
      !l.has_compatible_batch(r)) {
    THROW_ERROR(
        "Invalid shapes to calculate the matrix product: "
        << l.to_string() << ", " << r.to_string());
  }
  return Shape({l[0], r[1]}, std::max(l.batch(), r.batch()));
}

}  // namespace shape_ops
}  // namespace primitiv
