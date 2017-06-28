#include <config.h>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/shape_ops.h>

using std::vector;

// NOTE(odashi): This source only checks shape prerequisites of each operation.

#define CHECK_DEVICE(x) \
  if ((x).device() != this) { \
    THROW_ERROR( \
        "Device mismatched. (" #x ").device(): " << (x).device() \
        << "!= this: " << this); \
  }

namespace primitiv {

Tensor Device::new_tensor(const Shape &shape) {
  return Tensor(shape, this, new_handle(shape));
}

Tensor Device::new_tensor(const Shape &shape, float k) {
  Tensor ret(shape, this, new_handle(shape));
  reset_tensor(ret, k);
  return ret;
}

Tensor Device::new_tensor_by_array(const Shape &shape, const float values[]) {
  Tensor ret(shape, this, new_handle(shape));
  reset_tensor_by_array(ret, values);
  return ret;
}

Tensor Device::new_tensor_by_vector(
    const Shape &shape, const vector<float> &values) {
  Tensor ret(shape, this, new_handle(shape));
  reset_tensor_by_vector(ret, values);
  return ret;
}

vector<float> Device::tensor_to_vector(const Tensor &x) {
  CHECK_DEVICE(x);
  return tensor_to_vector_impl(x);
}

void Device::reset_tensor(Tensor &x, float k) {
  CHECK_DEVICE(x);
  reset_tensor_impl(x, k);
}

void Device::reset_tensor_by_array(Tensor &x, const float values[]) {
  // NOTE(odashi):
  // There is no method to guarantee the size of the array for now.
  CHECK_DEVICE(x);
  reset_tensor_by_array_impl(x, values);
}

void Device::reset_tensor_by_vector(Tensor &x, const vector<float> &values) {
  const unsigned num_elements = x.shape().num_total_elements();
  if (values.size() != num_elements) {
    THROW_ERROR(
        "Data sizes mismatched. required: " << num_elements
        << " (shape: " << x.shape().to_string() << ") != actual: "
        << values.size());
  }

  CHECK_DEVICE(x);
  reset_tensor_by_array_impl(x, values.data());
}

Tensor Device::copy_tensor(const Tensor &x) {
  if (!x.valid()) THROW_ERROR("Attempted to copy an invalid tensor.");
  return copy_tensor_impl(x);
}

Tensor Device::random_bernoulli(const Shape &shape, float p) {
  if (p < 0 || p > 1) {
    THROW_ERROR("Invalid Bernoulli probability: " << p);
  }
  return random_bernoulli_impl(shape, p);
}

Tensor Device::random_uniform(
    const Shape &shape, float lower, float upper) {
  if (upper < lower) {
    THROW_ERROR(
        "Invalid parameter of the uniform distribution. lower: " << lower
        << ", upper: " << upper);
  }
  return random_uniform_impl(shape, lower, upper);
}

Tensor Device::random_normal(const Shape &shape, float mean, float sd) {
  if (sd <= 0) {
    THROW_ERROR(
        "Invalid parameter of the normal distribution. mean: " << mean
        << ", SD: " << sd);
  }
  return random_normal_impl(shape, mean, sd);
}

Tensor Device::random_log_normal(const Shape &shape, float mean, float sd) {
  if (sd <= 0) {
    THROW_ERROR(
        "Invalid parameter of the log-normal distribution. mean: " << mean
        << ", SD: " << sd);
  }
  return random_log_normal_impl(shape, mean, sd);
}

Tensor Device::pick(
    const Tensor &x, unsigned dim, const vector<unsigned> &ids) {
  CHECK_DEVICE(x);
  return pick_impl(x, dim, ids, shape_ops::pick(x.shape(), dim, ids));
}

Tensor Device::slice(
    const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
  CHECK_DEVICE(x);
  return slice_impl(
      x, dim, lower, shape_ops::slice(x.shape(), dim, lower, upper));
}

Tensor Device::concat(const vector<const Tensor *> &xs, unsigned dim) {
  vector<const Shape *> shapes(xs.size());
  for (unsigned i = 0; i < xs.size(); ++i) {
    CHECK_DEVICE(*xs[i]);
    shapes[i] = &xs[i]->shape();
  }
  return concat_impl(xs, dim, shape_ops::concat(shapes, dim));
}

Tensor Device::negate(const Tensor &x) {
  CHECK_DEVICE(x);
  return negate_impl(x);
}

Tensor Device::add_const(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return add_const_impl(x, k);
}

Tensor Device::add_scalar(const Tensor &x, const Tensor &k) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(k);
  return add_scalar_impl(x, k, shape_ops::scalar_op(x.shape(), k.shape()));
}

Tensor Device::add(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return add_impl(a, b, shape_ops::elementwise(a.shape(), b.shape()));
}

Tensor Device::subtract_const_r(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return subtract_const_r_impl(x, k);
}

Tensor Device::subtract_const_l(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return subtract_const_l_impl(x, k);
}

Tensor Device::subtract_scalar_r(const Tensor &x, const Tensor &k) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(k);
  return subtract_scalar_r_impl(
      x, k, shape_ops::scalar_op(x.shape(), k.shape()));
}

Tensor Device::subtract_scalar_l(const Tensor &x, const Tensor &k) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(k);
  return subtract_scalar_l_impl(
      x, k, shape_ops::scalar_op(x.shape(), k.shape()));
}

Tensor Device::subtract(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return subtract_impl(a, b, shape_ops::elementwise(a.shape(), b.shape()));
}

Tensor Device::multiply_const(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return multiply_const_impl(x, k);
}

Tensor Device::multiply_scalar(const Tensor &x, const Tensor &k) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(k);
  return multiply_scalar_impl(x, k, shape_ops::scalar_op(x.shape(), k.shape()));
}

Tensor Device::multiply(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return multiply_impl(a, b, shape_ops::elementwise(a.shape(), b.shape()));
}

Tensor Device::divide_const_r(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return divide_const_r_impl(x, k);
}

Tensor Device::divide_const_l(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return divide_const_l_impl(x, k);
}

Tensor Device::divide_scalar_r(const Tensor &x, const Tensor &k) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(k);
  return divide_scalar_r_impl(x, k, shape_ops::scalar_op(x.shape(), k.shape()));
}

Tensor Device::divide_scalar_l(const Tensor &x, const Tensor &k) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(k);
  return divide_scalar_l_impl(x, k, shape_ops::scalar_op(x.shape(), k.shape()));
}

Tensor Device::divide(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return divide_impl(a, b, shape_ops::elementwise(a.shape(), b.shape()));
}

Tensor Device::transpose(const Tensor &x) {
  CHECK_DEVICE(x);
  return transpose_impl(x, shape_ops::transpose(x.shape()));
}

Tensor Device::dot(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return dot_impl(a, b, shape_ops::dot(a.shape(), b.shape()));
}

Tensor Device::sqrt(const Tensor &x) {
  CHECK_DEVICE(x);
  return sqrt_impl(x);
}

Tensor Device::exp(const Tensor &x) {
  CHECK_DEVICE(x);
  return exp_impl(x);
}

Tensor Device::tanh(const Tensor &x) {
  CHECK_DEVICE(x);
  return tanh_impl(x);
}

Tensor Device::sigmoid(const Tensor &x) {
  CHECK_DEVICE(x);
  return sigmoid_impl(x);
}

Tensor Device::pstep(const Tensor &x, float a) {
  if (a < 0 || a > 1) {
    THROW_ERROR("Parameter of 'pstep' should be in [0, 1]. a: " << a);
  }
  CHECK_DEVICE(x);
  return pstep_impl(x, a);
}

Tensor Device::prelu(const Tensor &x, float a) {
  if (a < 0 || a > 1) {
    THROW_ERROR("Parameter of 'prelu' should be in [0, 1]. a: " << a);
  }
  CHECK_DEVICE(x);
  return prelu_impl(x, a);
}

Tensor Device::sum(const Tensor &x, unsigned dim) {
  CHECK_DEVICE(x);
  return sum_impl(x, dim);
}

Tensor Device::logsumexp(const Tensor &x, unsigned dim) {
  CHECK_DEVICE(x);
  return logsumexp_impl(x, dim);
}

Tensor Device::broadcast(const Tensor &x, unsigned dim, unsigned size) {
  CHECK_DEVICE(x);
  return broadcast_impl(
      x, dim, size, shape_ops::broadcast(x.shape(), dim, size));
}

Tensor Device::batch_sum(const Tensor &x) {
  CHECK_DEVICE(x);
  return batch_sum_impl(x);
}

void Device::add_gradient(Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_dims(sb) || !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to add gradients with shape "
        << sb.to_string() << " to " << sa.to_string() << '.');
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  add_gradient_impl(a, b);
}

void Device::add_gradient_offset(
    Tensor &a, const Tensor &b, unsigned dim, unsigned offset) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_loo_dims(sb, dim) || !sa.has_compatible_batch(sb) ||
      offset + sb[dim] > sa[dim]) {
    THROW_ERROR(
        "Attempted to add gradients with shape "
        << sb.to_string() << ", dim " << dim << ", offset " << offset
        << " to shape" << sa.to_string() << '.');
  }

  if (dim >= sa.depth()) {
    add_gradient(a, b);
    return;
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  add_gradient_offset_impl(a, b, dim, offset);
}

void Device::add_gradient_sparse(
    Tensor &a, const Tensor &b,
    unsigned dim, const std::vector<unsigned> &ids) {
  const Shape sb = shape_ops::pick(a.shape(), dim, ids);
  if (sb != b.shape()) {
    THROW_ERROR(
        "Shape mismatched. b.shape(): " << b.shape().to_string()
        << " != expected shape: " << sb.to_string());
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  add_gradient_sparse_impl(a, b, dim, ids);
}

}  // namespace primitiv
