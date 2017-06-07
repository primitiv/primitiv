#include <config.h>

#include <primitiv/error.h>
#include <primitiv/shape.h>

using std::initializer_list;
using std::string;
using std::vector;

namespace primitiv {

Shape::Shape(const initializer_list<unsigned> &dims, const unsigned k)
: dims_(dims), k_(k) {
  adjust();
}

Shape::Shape(const vector<unsigned> &dims, const unsigned k)
: dims_(dims), k_(k) {
  adjust();
}

unsigned Shape::num_elements_under_rank(unsigned rank) const {
  unsigned ret = 1;
  for (unsigned i = 0; i < rank && i < depth(); ++i) ret *= dims_[i];
  return ret;
}

string Shape::to_string() const {
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

Shape Shape::resize_dim(unsigned dim, unsigned m) const {
  if (m == 0) THROW_ERROR("Could not set the dimension size 0.");
  Shape ret = *this;
  if (dim >= depth()) ret.dims_.resize(dim + 1, 1);
  ret.dims_[dim] = m;
  ret.adjust();
  return ret;
}

Shape Shape::resize_batch(unsigned k) const {
  if (k == 0) THROW_ERROR("Could not set the batch size 0.");
  Shape ret = *this;
  ret.k_ = k;
  return ret;
}

void Shape::adjust() {
  // erase redundant dimensions.
  while (!dims_.empty() && dims_.back() == 1) {
    dims_.pop_back();
  }

  // calculates the number of elements.
  num_elms_per_sample_ = 1;
  for (const unsigned n : dims_) num_elms_per_sample_ *= n;

  // check size of the shape.
  // if 1 or more dimensions or the batch size is 0,
  // then num_total_elements() returns 0.
  if (num_total_elements() == 0) {
    THROW_ERROR("invalid shape: " << to_string());
  }
}

}  // namespace primitiv
