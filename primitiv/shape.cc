#include <config.h>

#include <primitiv/error.h>
#include <primitiv/shape.h>

namespace primitiv {

Shape::Shape(std::initializer_list<unsigned> dims, unsigned batch)
: depth_(0), batch_(batch), volume_(1) {
  if (dims.size() > MAX_DEPTH) {
    THROW_ERROR(
        "Exceeds dimension depth limit at Shape::Shape()."
        " depth: " << depth_ << " > MAX_DEPTH: " << MAX_DEPTH);
  }
  for (const unsigned d : dims) {
    dims_[depth_++] = d;
    volume_ *= d;
  }
  while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
  if (volume_ == 0 || batch_ == 0) {
    THROW_ERROR("Invalid shape: " << to_string());
  }
}

Shape::Shape(const std::vector<unsigned> &dims, unsigned batch)
: depth_(0), batch_(batch), volume_(1) {
  if (dims.size() > MAX_DEPTH) {
    THROW_ERROR(
        "Exceeds dimension depth limit at Shape::Shape()."
        " depth: " << depth_ << " > MAX_DEPTH: " << MAX_DEPTH);
  }
  for (const unsigned d : dims) {
    dims_[depth_++] = d;
    volume_ *= d;
  }
  while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
  if (volume_ == 0 || batch_ == 0) {
    THROW_ERROR("Invalid shape: " << to_string());
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
  for (unsigned i = 0; i < depth_; ++i) {
    if (i > 0) {
      s << ',';
    }
    s << dims_[i];
  }
  s << "]x" << batch_;
  return s.str();
}

bool Shape::has_same_loo_dims(const Shape &rhs, unsigned dim) const {
  unsigned nl = depth_ == dim + 1 ? dim : depth_;
  while (nl > 0 && dims_[nl - 1] == 1) --nl;
  unsigned nr = rhs.depth_ == dim + 1 ? dim : rhs.depth_;
  while (nr > 0 && rhs.dims_[nr - 1] == 1) --nr;
  bool p = nl == nr;
  for (unsigned i = 0; i < nl; ++i) {
    p = p && (dims_[i] == rhs.dims_[i] || i == dim);
  }
  return p;
}

Shape Shape::resize_dim(unsigned dim, unsigned m) const {
  Shape ret = *this;
  ret.update_dim(dim, m);
  return ret;
}

Shape Shape::resize_batch(unsigned batch) const {
  Shape ret = *this;
  ret.update_batch(batch);
  return ret;
}

void Shape::update_dim(unsigned dim, unsigned m) {
  if (dim >= MAX_DEPTH) {
    THROW_ERROR(
      "Exceeds dimension depth limit at Shape::update_dim()."
      " dim: " << dim << " >= MAX_DEPTH: " << MAX_DEPTH);
  }
  if (m == 0) THROW_ERROR("Could not set each dimension to 0.");
  if (dim >= depth_) {
    unsigned new_depth = dim + 1;
    for (unsigned i = depth_; i < new_depth; ++i) dims_[i] = 1;
    depth_ = new_depth;
  }
  volume_ = (volume_ / dims_[dim]) * m;
  dims_[dim] = m;
  while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
}

void Shape::update_batch(unsigned batch) {
  if (batch == 0) THROW_ERROR("Could not set the batch size to 0.");
  batch_ = batch;
}

}  // namespace primitiv
