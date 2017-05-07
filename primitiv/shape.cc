#include <config.h>

#include <primitiv/shape.h>

#include <stdexcept>
#include <sstream>

using std::initializer_list;
using std::string;
using std::vector;

namespace primitiv {

Shape::Shape(const initializer_list<unsigned> &dim, const unsigned k)
: dim_(dim), k_(k) {
  // erase redundant dimensions.
  while (!dim_.empty() && dim_.back() == 1) {
    dim_.pop_back();
  }

  // check size of each dimension.
  for (const unsigned d : dim_) {
    if (d == 0) {
      throw std::runtime_error("invalid shape: " + to_string());
    }
  }

  if (k_ == 0) {
    throw std::runtime_error("invalid shape: " + to_string());
  }
}

string Shape::to_string() const {
  std::stringstream s;
  s << '[';
  for (unsigned i = 0; i < dim_.size(); ++i) {
    if (i > 0) {
      s << ',';
    }
    s << dim_[i];
  }
  s << "]x" << k_;
  return s.str();
}

}  // namespace primitiv
