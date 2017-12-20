#include <primitiv/config.h>

#include <primitiv/device.h>
#include <primitiv/functions.h>
#include <primitiv/parameter.h>

namespace {

using primitiv::Tensor;

// Helper to obtain Device object.
primitiv::Device &get_device(primitiv::Device *dev) {
  return dev ? *dev : primitiv::Device::get_default();
}

// Helper to transform tensors to pointers.
std::vector<const Tensor *> obj_to_ptr(const std::vector<Tensor> &xs) {
  std::vector<const Tensor *> ret;
  ret.reserve(xs.size());
  for (const Tensor &x : xs) ret.emplace_back(&x);
  return ret;
}

}  // namespace

namespace primitiv {

namespace functions {

template<>
Tensor positive(const Tensor &x) {
  return x;
}

template<>
Tensor negative(const Tensor &x) {
  return x.device().negate_fw(x);
}

template<>
Tensor add(const Tensor &x, float k) {
  return x.device().add_const_fw(x, k);
}

template<>
Tensor add(float k, const Tensor &x) {
  return x.device().add_const_fw(x, k);
}

template<>
Tensor add(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device().add_scalar_fw(b, a);
  else if (b.shape().is_scalar()) return a.device().add_scalar_fw(a, b);
  else return a.device().add_fw(a, b);
}

template<>
Tensor subtract(const Tensor &x, float k) {
  return x.device().subtract_const_r_fw(x, k);
}

template<>
Tensor subtract(float k, const Tensor &x) {
  return x.device().subtract_const_l_fw(x, k);
}

template<>
Tensor subtract(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device().subtract_scalar_l_fw(b, a);
  else if (b.shape().is_scalar()) return a.device().subtract_scalar_r_fw(a, b);
  else return a.device().subtract_fw(a, b);
}

template<>
Tensor multiply(const Tensor &x, float k) {
  return x.device().multiply_const_fw(x, k);
}

template<>
Tensor multiply(float k, const Tensor &x) {
  return x.device().multiply_const_fw(x, k);
}

template<>
Tensor multiply(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device().multiply_scalar_fw(b, a);
  else if (b.shape().is_scalar()) return a.device().multiply_scalar_fw(a, b);
  else return a.device().multiply_fw(a, b);
}

template<>
Tensor divide(const Tensor &x, float k) {
  return x.device().divide_const_r_fw(x, k);
}

template<>
Tensor divide(float k, const Tensor &x) {
  return x.device().divide_const_l_fw(x, k);
}

template<>
Tensor divide(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device().divide_scalar_l_fw(b, a);
  else if (b.shape().is_scalar()) return a.device().divide_scalar_r_fw(a, b);
  else return a.device().divide_fw(a, b);
}

template<>
Tensor pow(const Tensor &x, float k) {
  return x.device().pow_const_r_fw(x, k);
}

template<>
Tensor pow(float k, const Tensor &x) {
  return x.device().pow_const_l_fw(x, k);
}

template<>
Tensor pow(const Tensor &a, const Tensor &b) {
  if (a.shape().is_scalar()) return a.device().pow_scalar_l_fw(b, a);
  else if (b.shape().is_scalar()) return a.device().pow_scalar_r_fw(a, b);
  else return a.device().pow_fw(a, b);
}

Tensor input_tensor(
    const Shape &shape, const std::vector<float> &data, Device *dev) {
  return ::get_device(dev).new_tensor_by_vector(shape, data);
}

Tensor parameter_tensor(Parameter &param) {
  return param.value();
}

template<>
Tensor copy(const Tensor &x, Device *dev) {
  return ::get_device(dev).copy_tensor(x);
}

template<>
Tensor pick(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return x.device().pick_fw(x, ids, dim);
}

template<>
Tensor slice(const Tensor &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper) {
  return x.device().slice_fw(x, dim, lower, upper);
}

template<>
Tensor concat(const std::vector<const Tensor *> &xs, std::uint32_t dim) {
  if (xs.empty()) THROW_ERROR("No tensors to be concatenated.");
  return xs[0]->device().concat_fw(xs, dim);
}

template<>
Tensor concat(const std::vector<Tensor> &xs, std::uint32_t dim) {
  return concat(::obj_to_ptr(xs), dim);
}

template<>
Tensor reshape(const Tensor &x, const Shape &new_shape) {
  return x.reshape(new_shape);
}

template<>
Tensor flatten(const Tensor &x) {
  return x.flatten();
}

template<>
Tensor transpose(const Tensor &x) {
  return x.device().transpose_fw(x);
}

template<>
Tensor matmul(const Tensor &a, const Tensor &b) {
  return a.device().matmul_fw(a, b);
}

template<>
Tensor sqrt(const Tensor &x) {
  return x.device().sqrt_fw(x);
}

template<>
Tensor exp(const Tensor &x) {
  return x.device().exp_fw(x);
}

template<>
Tensor log(const Tensor &x) {
  return x.device().log_fw(x);
}

template<>
Tensor tanh(const Tensor &x) {
  return x.device().tanh_fw(x);
}

template<>
Tensor sigmoid(const Tensor &x) {
  return x.device().sigmoid_fw(x);
}

template<>
Tensor softplus(const Tensor &x) {
  return x.device().softplus_fw(x);
}

template<>
Tensor sin(const Tensor &x) {
  return x.device().sin_fw(x);
}

template<>
Tensor cos(const Tensor &x) {
  return x.device().cos_fw(x);
}

template<>
Tensor tan(const Tensor &x) {
  return x.device().tan_fw(x);
}

template<>
Tensor relu(const Tensor &x) {
  return x.device().prelu_fw(x, 0);
}

template<>
Tensor lrelu(const Tensor &x) {
  return x.device().prelu_fw(x, .01);
}

template<>
Tensor prelu(const Tensor &x, float a) {
  return x.device().prelu_fw(x, a);
}

template<>
Tensor elu(const Tensor &x, float a) {
  return x.device().elu_fw(x, a);
}

template<>
Tensor sum(const Tensor &x, std::uint32_t dim) {
  return x.device().sum_fw(x, dim);
}

template<>
Tensor broadcast(const Tensor &x, std::uint32_t dim, std::uint32_t size) {
  return x.device().broadcast_fw(x, dim, size);
}

template<>
Tensor logsumexp(const Tensor &x, std::uint32_t dim) {
  return x.device().logsumexp_fw(x, dim);
}

template<>
Tensor log_softmax(const Tensor &x, std::uint32_t dim) {
  return x - broadcast(logsumexp(x, dim), dim, x.shape()[dim]);
}

template<>
Tensor softmax(const Tensor &x, std::uint32_t dim) {
  return exp(log_softmax(x, dim));
}

template<>
Tensor softmax_cross_entropy(const Tensor &x, const Tensor &t, std::uint32_t dim) {
  return -sum(t * log_softmax(x, dim), dim);
}

template<>
Tensor softmax_cross_entropy(
    const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim) {
  return pick(-log_softmax(x, dim), ids, dim);
}

template<>
Tensor stop_gradient(const Tensor &x) { return x; }

namespace batch {

template<>
Tensor sum(const Tensor &x) {
  return x.device().batch_sum_fw(x);
}

}  // namespace batch

Tensor constant_tensor(const Shape &shape, float k, Device *dev) {
  return ::get_device(dev).new_tensor_by_constant(shape, k);
}

Tensor identity_tensor(std::uint32_t size, Device *dev) {
  return ::get_device(dev).identity(size);
}

namespace random {

Tensor bernoulli_tensor(
    const Shape &shape, float p, Device *dev) {
  return ::get_device(dev).random_bernoulli(shape, p);
}

Tensor uniform_tensor(
    const Shape &shape, float lower, float upper, Device *dev) {
  return ::get_device(dev).random_uniform(shape, lower, upper);
}

Tensor normal_tensor(
    const Shape &shape, float mean, float sd, Device *dev) {
  return ::get_device(dev).random_normal(shape, mean, sd);
}

Tensor log_normal_tensor(
    const Shape &shape, float mean, float sd, Device *dev) {
  return ::get_device(dev).random_log_normal(shape, mean, sd);
}

Tensor gumbel_tensor(
    const Shape &shape, float mu, float beta, Device *dev) {
  return mu - beta * log(-log(uniform_tensor(shape, 0., .9999999, dev)));
}

}  // namespace random

}  // namespace functions

}  // namespace primitiv
