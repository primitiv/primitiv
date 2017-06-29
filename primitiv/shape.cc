#include <config.h>

#include <primitiv/error.h>
#include <primitiv/shape.h>

namespace primitiv {

Shape &Shape::operator=(Shape &&src) {
  if (&src != this) {
    dims_ = std::move(src.dims_);
    k_ = src.k_;
    volume_ = src.volume_;
  }
  return *this;
}

unsigned Shape::lower_volume(unsigned rank) const {
  unsigned ret = 1;
  for (unsigned i = 0; i < rank && i < depth(); ++i) ret *= dims_[i];
  return ret;
}

std::string Shape::to_string() const {
  std::stringstream s;
  s << '[';
  for (unsigned i = 0; i < depth(); ++i) {
    if (i > 0) {
      s << ',';
    }
    s << dims_[i];
  }
  s << "]x" << k_;
  return s.str();
}

bool Shape::has_same_loo_dims(const Shape &rhs, unsigned dim) const {
  unsigned nl = dims_.size() == dim + 1 ? dim : dims_.size();
  while (nl > 0 && dims_[nl - 1] == 1) --nl;
  unsigned nr = rhs.dims_.size() == dim + 1 ? dim : rhs.dims_.size();
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

Shape Shape::resize_batch(unsigned k) const {
  Shape ret = *this;
  ret.update_batch(k);
  return ret;
}

void Shape::update_dim(unsigned dim, unsigned m) {
  if (m == 0) THROW_ERROR("Could not set the dimension size 0.");
  if (dim >= depth()) dims_.resize(dim + 1, 1);
  dims_[dim] = m;
  adjust();
}

void Shape::update_batch(unsigned k) {
  if (k == 0) THROW_ERROR("Could not set the batch size 0.");
  k_ = k;
}

void Shape::adjust() {
  // erase redundant dimensions.
  while (!dims_.empty() && dims_.back() == 1) {
    dims_.pop_back();
  }

  // calculates the number of elements.
  volume_ = 1;
  for (const unsigned n : dims_) volume_ *= n;

  // check size of the shape.
  // if 1 or more dimensions or the batch size is 0,
  // size() returns 0.
  if (size() == 0) {
    THROW_ERROR("invalid shape: " << to_string());
  }
}

}  // namespace primitiv
