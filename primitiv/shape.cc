#include <config.h>

#include <primitiv/error.h>
#include <primitiv/shape.h>

using std::initializer_list;
using std::string;
using std::vector;

namespace primitiv {

Shape::Shape(const initializer_list<unsigned> &dims, const unsigned k)
: dims_(dims), k_(k), size_per_sample_(1) {
  adjust();
}

Shape::Shape(const vector<unsigned> &dims, const unsigned k)
: dims_(dims), k_(k), size_per_sample_(1) {
  adjust();
}

string Shape::to_string() const {
  std::stringstream s;
  s << '[';
  for (unsigned i = 0; i < dims_.size(); ++i) {
    if (i > 0) {
      s << ',';
    }
    s << dims_[i];
  }
  s << "]x" << k_;
  return s.str();
}

void Shape::adjust() {
  // erase redundant dimensions.
  while (!dims_.empty() && dims_.back() == 1) {
    dims_.pop_back();
  }

  // calculates the number of elements.
  for (const unsigned n : dims_) size_per_sample_ *= n;

  // check size of the shape.
  // if 1 or more dimensions or the batch size is 0, then size() returns 0.
  if (size() == 0) {
    THROW_ERROR("invalid shape: " << to_string());
  }
}

}  // namespace primitiv
