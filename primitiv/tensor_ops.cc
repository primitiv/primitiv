#include <config.h>

#include <primitiv/device.h>
#include <primitiv/tensor_ops.h>

namespace primitiv {

Tensor operator+(const Tensor &x) {
  return x.device()->duplicate(x);
}

Tensor operator-(const Tensor &x) {
  return x.device()->negate(x);\
}

Tensor operator+(const Tensor &x, float k) {
  return x.device()->add(x, k);
}

Tensor operator+(float k, const Tensor &x) {
  return x.device()->add(x, k);
}

Tensor operator+(const Tensor &a, const Tensor &b) {
  return a.device()->add(a, b);
}

Tensor operator-(const Tensor &x, float k) {
  return x.device()->subtract(x, k);
}

Tensor operator-(float k, const Tensor &x) {
  return x.device()->subtract(k, x);
}

Tensor operator-(const Tensor &a, const Tensor &b) {
  return a.device()->subtract(a, b);
}

Tensor operator*(const Tensor &x, float k) {
  return x.device()->multiply(x, k);
}

Tensor operator*(float k, const Tensor &x) {
  return x.device()->multiply(x, k);
}

Tensor operator*(const Tensor &a, const Tensor &b) {
  return a.device()->multiply(a, b);
}

Tensor operator/(const Tensor &x, float k) {
  return x.device()->divide(x, k);
}

Tensor operator/(float k, const Tensor &x) {
  return x.device()->divide(k, x);
}

Tensor operator/(const Tensor &a, const Tensor &b) {
  return a.device()->divide(a, b);
}

namespace tensor_ops {

Tensor slice(const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
  return x.device()->slice(x, dim, lower, upper);
}
Tensor concat(const std::vector<const Tensor *> &xs, unsigned dim) {
  if (xs.empty()) THROW_ERROR("No tensors to be concatenated.");
  return xs[0]->device()->concat(xs, dim);
}

Tensor transpose(const Tensor &x) {
  return x.device()->transpose(x);
}

Tensor dot(const Tensor &a, const Tensor &b) {
  return a.device()->dot(a, b);
}

Tensor exp(const Tensor &x) {
  return x.device()->exp(x);
}

Tensor tanh(const Tensor &x) {
  return x.device()->tanh(x);
}

Tensor sigmoid(const Tensor &x) {
  return x.device()->sigmoid(x);
}

Tensor step(const Tensor &x) {
  return x.device()->step(x);
}

Tensor relu(const Tensor &x) {
  return x.device()->relu(x);
}

Tensor sum(const Tensor &x, unsigned dim) {
  return x.device()->sum(x, dim);
}

Tensor batch_sum(const Tensor &x) {
  return x.device()->batch_sum(x);
}

}  // namespace tensor_ops
}  // namespace primitiv
