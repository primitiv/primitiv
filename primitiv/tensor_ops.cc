#include <config.h>

#include <primitiv/device.h>
#include <primitiv/tensor_ops.h>

namespace primitiv {

Tensor operator+(const Tensor &x) {
  return x;
}

Tensor operator-(const Tensor &x) {
  return x.device()->negate(x);
}

Tensor operator+(const Tensor &x, float k) {
  return x.device()->add_const(x, k);
}

Tensor operator+(float k, const Tensor &x) {
  return x.device()->add_const(x, k);
}

Tensor operator+(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->add_scalar(b, a);
  else if (b.shape().is_scalar()) return a.device()->add_scalar(a, b);
  else return a.device()->add(a, b);
}

Tensor operator-(const Tensor &x, float k) {
  return x.device()->subtract_const_r(x, k);
}

Tensor operator-(float k, const Tensor &x) {
  return x.device()->subtract_const_l(x, k);
}

Tensor operator-(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->subtract_scalar_l(b, a);
  else if (b.shape().is_scalar()) return a.device()->subtract_scalar_r(a, b);
  else return a.device()->subtract(a, b);
}

Tensor operator*(const Tensor &x, float k) {
  return x.device()->multiply_const(x, k);
}

Tensor operator*(float k, const Tensor &x) {
  return x.device()->multiply_const(x, k);
}

Tensor operator*(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->multiply_scalar(b, a);
  else if (b.shape().is_scalar()) return a.device()->multiply_scalar(a, b);
  else return a.device()->multiply(a, b);
}

Tensor operator/(const Tensor &x, float k) {
  return x.device()->divide_const_r(x, k);
}

Tensor operator/(float k, const Tensor &x) {
  return x.device()->divide_const_l(x, k);
}

Tensor operator/(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->divide_scalar_l(b, a);
  else if (b.shape().is_scalar()) return a.device()->divide_scalar_r(a, b);
  else return a.device()->divide(a, b);
}

namespace tensor_ops {

Tensor copy(const Tensor &x, Device *dev) {
  return dev->copy_tensor(x);
}

Tensor pick(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids) {
  return x.device()->pick(x, dim, ids);
}

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

Tensor sqrt(const Tensor &x) {
  return x.device()->sqrt(x);
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

Tensor logsumexp(const Tensor &x, unsigned dim) {
  return x.device()->logsumexp(x, dim);
}

Tensor log_softmax(const Tensor &x, unsigned dim) {
  return x - broadcast(logsumexp(x, dim), dim, x.shape()[dim]);
}

Tensor softmax(const Tensor &x, unsigned dim) {
  return exp(log_softmax(x, dim));
}

Tensor broadcast(const Tensor &x, unsigned dim, unsigned size) {
  return x.device()->broadcast(x, dim, size);
}

Tensor batch_sum(const Tensor &x) {
  return x.device()->batch_sum(x);
}

Tensor softmax_cross_entropy(const Tensor &x, const Tensor &t, unsigned dim) {
  return -sum(t * log_softmax(x, dim), dim);
}

}  // namespace tensor_ops
}  // namespace primitiv
