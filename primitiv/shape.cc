#include <config.h>

#include <primitiv/shape.h>

#include <stdexcept>
#include <sstream>

using std::initializer_list;
using std::string;
using std::vector;

namespace primitiv {

Shape::Shape(const initializer_list<unsigned> dim, const unsigned k)
: dim_(dim), k_(k) {
  adjust();
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

void Shape::adjust() {
  // erase redundant dimensions.
  while (!dim_.empty() && dim_.back() == 1) {
    dim_.pop_back();
  }

  // check size of the shape.
  // if more than 1 dimensions or the batch size is 0, then size() returns 0.
  if (size() == 0) {
    throw std::runtime_error("invalid shape: " + to_string());
  }
}

}  // namespace primitiv
