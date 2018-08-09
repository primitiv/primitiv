#include <primitiv/config.h>

#include <algorithm>

#include <primitiv/core/error.h>
#include <primitiv/core/shape_ops.h>

namespace primitiv {
namespace shape_ops {

Shape reshape(const Shape &before, const Shape &after) {
  if (before.volume() != after.volume() ||
      (after.has_batch() && after.batch() != before.batch())) {
    PRIMITIV_THROW_ERROR(
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
    PRIMITIV_THROW_ERROR(
        "Shape mismatched for the scalar operation. "
        "x: " << x.to_string() << " != k: " << k.to_string());
  }
  return x.resize_batch(std::max(x.batch(), k.batch()));
}

Shape elementwise(const Shape &a, const Shape &b) {
  if (!a.has_same_dims(b) || !a.has_compatible_batch(b)) {
    PRIMITIV_THROW_ERROR(
        "Shape mismatched for the elementwise operation. "
        "a: " << a.to_string() << " != b: " << b.to_string());
  }
  return a.resize_batch(std::max(a.batch(), b.batch()));
}

Shape slice(
    const Shape &x, std::uint32_t dim,
    std::uint32_t lower, std::uint32_t upper) {
  if (lower >= upper || upper > x[dim]) {
    PRIMITIV_THROW_ERROR(
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
    PRIMITIV_THROW_ERROR("No tensors to be concatenated.");
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
      PRIMITIV_THROW_ERROR("Invalid shapes to concatenate: " << dims_str);
    }
    if (!s0.has_batch()) s0.update_batch(s.batch());
    sum += s[dim];
  }

  s0.update_dim(dim, sum);
  return s0;
}

Shape broadcast(const Shape &x, std::uint32_t dim, std::uint32_t size) {
  if (x[dim] != 1 || size == 0) {
    PRIMITIV_THROW_ERROR(
        "Invalid broadcasting. x: "
        << x.to_string() << ", dim: " << dim << ", size: " << size);
  }
  return x.resize_dim(dim, size);
}

Shape pick(
    const Shape &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  const std::uint32_t n = x[dim];
  const std::uint32_t bi = ids.size();
  if (bi == 0 || (x.batch() != bi && x.has_batch() && bi > 1)) {
    PRIMITIV_THROW_ERROR(
        "Invalid IDs to pick. shape: " << x.to_string()
        << ", ids.size(): " << ids.size());
  }
  for (std::uint32_t i = 0; i < bi; ++i) {
    if (ids[i] >= n) {
      PRIMITIV_THROW_ERROR(
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
    PRIMITIV_THROW_ERROR("Invalid shape to transpose: " << x.to_string());
  }
  return Shape({x[1], x[0]}, x.batch());
}

Shape permute_dims(const Shape &x, const std::vector<std::uint32_t> &perm) {
  if (perm.size() < x.depth()) {
    PRIMITIV_THROW_ERROR(
        "Invalid perm to permute. shape: " << x.to_string()
        << ", perm.size(): " << perm.size());
  }
  std::vector<std::uint32_t> dims(perm.size());
  std::vector<bool> picked(perm.size());
  for (std::uint32_t i = 0; i < perm.size(); ++i) {
    if (perm[i] >= perm.size()) {
      PRIMITIV_THROW_ERROR(
          "Invalid perm to permute. shape: " << x.to_string()
          << ", perm.size(): " << perm.size()
          << ", perm[" << i << "]: " << perm[i]);
    }
    if (picked[perm[i]]) {
      PRIMITIV_THROW_ERROR(
          "Invalid perm to permute. shape: " << x.to_string()
          << ", perm[" << i << "]: " << perm[i]
          << " (duplicated)");
    }
    picked[perm[i]] = true;
    dims[i] = x[perm[i]];
  }
  return Shape(dims, x.batch());
}

Shape matmul(const Shape &l, const Shape &r) {
  if (!l.is_matrix() || !r.is_matrix() || l[1] != r[0] ||
      !l.has_compatible_batch(r)) {
    PRIMITIV_THROW_ERROR(
        "Invalid shapes to calculate the matrix product: "
        << l.to_string() << ", " << r.to_string());
  }
  return Shape({l[0], r[1]}, std::max(l.batch(), r.batch()));
}

Shape conv2d(
    const Shape &x, const Shape &w,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1) {
  const std::uint32_t x0 = x[0] + 2 * padding0;
  const std::uint32_t x1 = x[1] + 2 * padding1;
  const std::uint32_t w0 = (w[0] - 1) * dilation0 + 1;
  const std::uint32_t w1 = (w[1] - 1) * dilation1 + 1;

  if (x.depth() > 3 || w.depth() > 4 ||
      x0 < w0 || x1 < w1 || x[2] != w[2] ||
      !x.has_compatible_batch(w) ||
      stride0 == 0 || stride1 == 0 ||
      dilation0 == 0 || dilation1 == 0) {
    PRIMITIV_THROW_ERROR(
        "Invalid arguments to calculate a convolution: "
        << x.to_string() << ", " << w.to_string() << ", "
        << padding0 << ", " << padding1 << ", "
        << stride0 << ", " << stride1 << ", "
        << dilation0 << ", " << dilation1);
  }
  return Shape(
      {(x0 - w0) / stride0 + 1, (x1 - w1) / stride1 + 1, w[3]},
      std::max(x.batch(), w.batch()));
}

Shape pool2d(
    const Shape &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1) {
  const std::uint32_t x0 = x[0] + 2 * padding0;
  const std::uint32_t x1 = x[1] + 2 * padding1;

  if (x.depth() > 3 ||
      x0 < window0 || x1 < window1 ||
      window0 == 0 || window1 == 0 ||
      stride0 == 0 || stride1 == 0) {
    PRIMITIV_THROW_ERROR(
        "Invalid arguments to calculate a pooling: "
        << x.to_string() << ", "
        << window0 << ", " << window1 << ", "
        << padding0 << ", " << padding1 << ", "
        << stride0 << ", " << stride1);
  }

  return Shape(
      {(x0 - window0) / stride0 + 1, (x1 - window1) / stride1 + 1, x[2]},
      x.batch());
}

Shape batch_pick(const Shape &x, const std::vector<std::uint32_t> &ids) {
  const std::uint32_t n = x.batch();
  const std::uint32_t bi = ids.size();
  if (bi == 0) {
    PRIMITIV_THROW_ERROR(
        "Invalid IDs to pick. shape: " << x.to_string()
        << ", ids.size(): " << ids.size());
  }
  for (std::uint32_t i = 0; i < bi; ++i) {
    if (ids[i] >= n) {
      PRIMITIV_THROW_ERROR(
          "Invalid IDs to pick. shape: " << x.to_string()
          << ", ids[" << i << "]: " << ids[i]);
    }
  }

  return x.resize_batch(bi);
}

Shape batch_slice(const Shape &x, std::uint32_t lower, std::uint32_t upper) {
  if (lower >= upper || upper > x.batch()) {
    PRIMITIV_THROW_ERROR(
        "Invalid batch_slice operation. shape: " << x.to_string()
        << ", lower: " << lower << ", upper: " << upper);
  }

  return x.resize_batch(upper - lower);
}

Shape batch_concat(const std::vector<Shape> &xs) {
  std::vector<const Shape *> ptrs;
  ptrs.reserve(xs.size());
  for (const Shape &x : xs) ptrs.emplace_back(&x);
  return batch_concat(ptrs);
}

Shape batch_concat(const std::vector<const Shape *> &xs) {
  if (xs.empty()) {
    PRIMITIV_THROW_ERROR("No tensors to be concatenated.");
  }

  Shape s0 = *xs[0];
  std::uint32_t sum = s0.batch();

  for (std::uint32_t i = 1; i < xs.size(); ++i) {
    const Shape &s = *xs[i];
    if (!s0.has_same_dims(s)) {
      std::string shapes_str = xs[0]->to_string();
      for (std::uint32_t i = 1; i < xs.size(); ++i) {
        shapes_str += ", " + xs[i]->to_string();
      }
      PRIMITIV_THROW_ERROR("Invalid shapes to concatenate minibatches: " << shapes_str);
    }
    sum += s.batch();
  }

  s0.update_batch(sum);
  return s0;
}

}  // namespace shape_ops
}  // namespace primitiv
