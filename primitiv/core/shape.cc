#include <primitiv/config.h>

#include <primitiv/core/error.h>
#include <primitiv/core/shape.h>

namespace primitiv {

Shape::Shape(std::initializer_list<std::uint32_t> dims, std::uint32_t batch)
: depth_(0), batch_(batch), volume_(1) {
  if (dims.size() > MAX_DEPTH) {
    PRIMITIV_THROW_ERROR(
        "Exceeds dimension depth limit at Shape::Shape()."
        " depth: " << depth_ << " > MAX_DEPTH: " << MAX_DEPTH);
  }
  for (const std::uint32_t d : dims) {
    dims_[depth_++] = d;
    volume_ *= d;
  }
  while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
  if (volume_ == 0 || batch_ == 0) {
    PRIMITIV_THROW_ERROR("Invalid shape: " << to_string());
  }
}

Shape::Shape(const std::vector<std::uint32_t> &dims, std::uint32_t batch)
: depth_(0), batch_(batch), volume_(1) {
  if (dims.size() > MAX_DEPTH) {
    PRIMITIV_THROW_ERROR(
        "Exceeds dimension depth limit at Shape::Shape()."
        " depth: " << depth_ << " > MAX_DEPTH: " << MAX_DEPTH);
  }
  for (const std::uint32_t d : dims) {
    dims_[depth_++] = d;
    volume_ *= d;
  }
  while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
  if (volume_ == 0 || batch_ == 0) {
    PRIMITIV_THROW_ERROR("Invalid shape: " << to_string());
  }
}

Shape &Shape::operator=(Shape &&src) {
  if (&src != this) {
    dims_ = std::move(src.dims_);
    depth_ = src.depth_;
    batch_ = src.batch_;
    volume_ = src.volume_;
  }
  return *this;
}

std::string Shape::to_string() const {
  std::stringstream s;
  s << '[';
  for (std::uint32_t i = 0; i < depth_; ++i) {
    if (i > 0) {
      s << ',';
    }
    s << dims_[i];
  }
  s << "]x" << batch_;
  return s.str();
}

bool Shape::has_same_loo_dims(const Shape &rhs, std::uint32_t dim) const {
  std::uint32_t nl = depth_ == dim + 1 ? dim : depth_;
  while (nl > 0 && dims_[nl - 1] == 1) --nl;
  std::uint32_t nr = rhs.depth_ == dim + 1 ? dim : rhs.depth_;
  while (nr > 0 && rhs.dims_[nr - 1] == 1) --nr;
  bool p = nl == nr;
  for (std::uint32_t i = 0; i < nl; ++i) {
    p = p && (dims_[i] == rhs.dims_[i] || i == dim);
  }
  return p;
}

Shape Shape::resize_dim(std::uint32_t dim, std::uint32_t m) const {
  Shape ret = *this;
  ret.update_dim(dim, m);
  return ret;
}

Shape Shape::resize_batch(std::uint32_t batch) const {
  Shape ret = *this;
  ret.update_batch(batch);
  return ret;
}

void Shape::update_dim(std::uint32_t dim, std::uint32_t m) {
  if (dim >= MAX_DEPTH) {
    PRIMITIV_THROW_ERROR(
      "Exceeds dimension depth limit at Shape::update_dim()."
      " dim: " << dim << " >= MAX_DEPTH: " << MAX_DEPTH);
  }
  if (m == 0) PRIMITIV_THROW_ERROR("Could not set each dimension to 0.");
  if (dim >= depth_) {
    std::uint32_t new_depth = dim + 1;
    for (std::uint32_t i = depth_; i < new_depth; ++i) dims_[i] = 1;
    depth_ = new_depth;
  }
  volume_ = (volume_ / dims_[dim]) * m;
  dims_[dim] = m;
  while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
}

void Shape::update_batch(std::uint32_t batch) {
  if (batch == 0) PRIMITIV_THROW_ERROR("Could not set the batch size to 0.");
  batch_ = batch;
}

}  // namespace primitiv
