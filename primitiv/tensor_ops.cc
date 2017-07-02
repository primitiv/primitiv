#include <config.h>

#include <primitiv/device.h>
#include <primitiv/tensor_ops.h>

namespace primitiv {

Tensor operator+(const Tensor &x) {
  return x;
}

Tensor operator-(const Tensor &x) {
  return x.device()->negate_fw(x);
}

Tensor operator+(const Tensor &x, float k) {
  return x.device()->add_const_fw(x, k);
}

Tensor operator+(float k, const Tensor &x) {
  return x.device()->add_const_fw(x, k);
}

Tensor operator+(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->add_scalar_fw(b, a);
  else if (b.shape().is_scalar()) return a.device()->add_scalar_fw(a, b);
  else return a.device()->add_fw(a, b);
}

Tensor operator-(const Tensor &x, float k) {
  return x.device()->subtract_const_r_fw(x, k);
}

Tensor operator-(float k, const Tensor &x) {
  return x.device()->subtract_const_l_fw(x, k);
}

Tensor operator-(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->subtract_scalar_l_fw(b, a);
  else if (b.shape().is_scalar()) return a.device()->subtract_scalar_r_fw(a, b);
  else return a.device()->subtract_fw(a, b);
}

Tensor operator*(const Tensor &x, float k) {
  return x.device()->multiply_const_fw(x, k);
}

Tensor operator*(float k, const Tensor &x) {
  return x.device()->multiply_const_fw(x, k);
}

Tensor operator*(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->multiply_scalar_fw(b, a);
  else if (b.shape().is_scalar()) return a.device()->multiply_scalar_fw(a, b);
  else return a.device()->multiply_fw(a, b);
}

Tensor operator/(const Tensor &x, float k) {
  return x.device()->divide_const_r_fw(x, k);
}

Tensor operator/(float k, const Tensor &x) {
  return x.device()->divide_const_l_fw(x, k);
}

Tensor operator/(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device()->divide_scalar_l_fw(b, a);
  else if (b.shape().is_scalar()) return a.device()->divide_scalar_r_fw(a, b);
  else return a.device()->divide_fw(a, b);
}

namespace tensor_ops {

Tensor copy(const Tensor &x, Device *dev) {
  return dev->copy_tensor(x);
}

Tensor pick(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids) {
  return x.device()->pick_fw(x, dim, ids);
}

Tensor slice(const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
  return x.device()->slice_fw(x, dim, lower, upper);
}

Tensor concat(const std::vector<const Tensor *> &xs, unsigned dim) {
  if (xs.empty()) THROW_ERROR("No tensors to be concatenated.");
  return xs[0]->device()->concat_fw(xs, dim);
}

Tensor reshape(const Tensor &x, const Shape &new_shape) {
  return x.reshape(new_shape);
}

Tensor flatten(const Tensor &x) {
  return x.flatten();
}

Tensor transpose(const Tensor &x) {
  return x.device()->transpose_fw(x);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  return a.device()->matmul_fw(a, b);
}

Tensor sqrt(const Tensor &x) {
  return x.device()->sqrt_fw(x);
}

Tensor exp(const Tensor &x) {
  return x.device()->exp_fw(x);
}

Tensor tanh(const Tensor &x) {
  return x.device()->tanh_fw(x);
}

Tensor sigmoid(const Tensor &x) {
  return x.device()->sigmoid_fw(x);
}

Tensor sin(const Tensor &x) {
  return x.device()->sin_fw(x);
}

Tensor cos(const Tensor &x) {
  return x.device()->cos_fw(x);
}

Tensor tan(const Tensor &x) {
  return x.device()->tan_fw(x);
}

Tensor relu(const Tensor &x) {
  return x.device()->prelu_fw(x, 0);
}

Tensor lrelu(const Tensor &x) {
  return x.device()->prelu_fw(x, .01);
}

Tensor prelu(const Tensor &x, float a) {
  return x.device()->prelu_fw(x, a);
}

Tensor sum(const Tensor &x, unsigned dim) {
  return x.device()->sum_fw(x, dim);
}

Tensor logsumexp(const Tensor &x, unsigned dim) {
  return x.device()->logsumexp_fw(x, dim);
}

Tensor log_softmax(const Tensor &x, unsigned dim) {
  return x - broadcast(logsumexp(x, dim), dim, x.shape()[dim]);
}

Tensor softmax(const Tensor &x, unsigned dim) {
  return exp(log_softmax(x, dim));
}

Tensor broadcast(const Tensor &x, unsigned dim, unsigned size) {
  return x.device()->broadcast_fw(x, dim, size);
}

Tensor batch_sum(const Tensor &x) {
  return x.device()->batch_sum_fw(x);
}

Tensor softmax_cross_entropy(const Tensor &x, const Tensor &t, unsigned dim) {
  return -sum(t * log_softmax(x, dim), dim);
}

}  // namespace tensor_ops
}  // namespace primitiv
