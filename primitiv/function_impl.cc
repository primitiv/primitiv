#include <config.h>

#include <sstream>
#include <stdexcept>
#include <primitiv/function_impl.h>

namespace primitiv {
namespace functions {

#define CHECK_ARGNUM(name, args, n) \
  if (args.size() != n) { \
    std::stringstream ss; \
    ss << "Number of arguments mismatched." \
       << " function: " << #name \
       << ", expected: " << n \
       << " != actual: " << args.size(); \
    throw std::runtime_error(ss.str()); \
  }

Input::Input(const Shape &shape) : shape_(shape) {}

Shape Input::forward_shape(const std::vector<const Shape *> &args) const {
  CHECK_ARGNUM(Input, args, 0);
  return shape_;
}

Tensor Input::forward(const std::vector<const Tensor *> &args) const {
  CHECK_ARGNUM(Input, args, 0);
  throw 123;
}

}  // namespace functions
}  // namespace primitive
