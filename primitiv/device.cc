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

Tensor Device::new_tensor(
    const Shape &shape, const vector<float> &values) {
  Tensor ret(shape, this, new_handle(shape));
  reset_tensor(ret, values);
  return ret;
}

void Device::delete_tensor(Tensor &x) {
  delete_tensor_impl(x);
}

vector<float> Device::tensor_to_vector(const Tensor &x) {
  CHECK_DEVICE(x);
  return tensor_to_vector_impl(x);
}

void Device::reset_tensor(Tensor &x, float k) {
  CHECK_DEVICE(x);
  reset_tensor_impl(x, k);
}

void Device::reset_tensor(Tensor &x, const vector<float> &values) {
  const unsigned num_elements = x.shape().num_total_elements();
  if (values.size() != num_elements) {
    THROW_ERROR(
        "Data sizes mismatched. required: " << num_elements
        << " (shape: " << x.shape().to_string() << ") != actual: "
        << values.size());
  }

  CHECK_DEVICE(x);
  reset_tensor_impl(x, values);
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

Tensor Device::slice(
    const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
  const Shape new_shape = shape_ops::slice(x.shape(), dim, lower, upper);

  CHECK_DEVICE(x);
  return slice_impl(x, dim, lower, new_shape);
}

Tensor Device::concat(const vector<const Tensor *> &xs, unsigned dim) {
  vector<const Shape *> shapes(xs.size());
  for (unsigned i = 0; i < xs.size(); ++i) shapes[i] = &xs[i]->shape();
  const Shape new_shape = shape_ops::concat(shapes, dim);

  for (const Tensor *x : xs) CHECK_DEVICE(*x);
  return concat_impl(xs, dim, new_shape);
}

Tensor Device::duplicate(const Tensor &x) {
  CHECK_DEVICE(x);
  return duplicate_impl(x);
}

Tensor Device::negate(const Tensor &x) {
  CHECK_DEVICE(x);
  return negate_impl(x);
}

Tensor Device::add(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return add_impl(x, k);
}

Tensor Device::add(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_dims(sb) || !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to add tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return add_impl(a, b);
}

Tensor Device::subtract(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return subtract_impl(x, k);
}

Tensor Device::subtract(float k, const Tensor &x) {
  CHECK_DEVICE(x);
  return subtract_impl(k, x);
}

Tensor Device::subtract(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_dims(sb) || !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to subtract tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return subtract_impl(a, b);
}

Tensor Device::multiply(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return multiply_impl(x, k);
}

Tensor Device::multiply(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_dims(sb) || !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to multiply tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return multiply_impl(a, b);
}

Tensor Device::divide(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return divide_impl(x, k);
}

Tensor Device::divide(float k, const Tensor &x) {
  CHECK_DEVICE(x);
  return divide_impl(k, x);
}

Tensor Device::divide(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_dims(sb) || !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to divide tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return divide_impl(a, b);
}

Tensor Device::transpose(const Tensor &x) {
  const Shape &s = x.shape();
  if (s.depth() > 2) {
    THROW_ERROR(
        "Attempted to transpose a tensor with shape " << s.to_string() << '.');
  }

  CHECK_DEVICE(x);
  return transpose_impl(x);
}

Tensor Device::dot(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (sa.depth() > 2 || sb.depth() > 2 || sa[1] != sb[0] ||
      !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to calculate the dot product of tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }

  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return dot_impl(a, b);
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

Tensor Device::step(const Tensor &x) {
  CHECK_DEVICE(x);
  return step_impl(x);
}

Tensor Device::relu(const Tensor &x) {
  CHECK_DEVICE(x);
  return relu_impl(x);
}

Tensor Device::sum(const Tensor &x, unsigned dim) {
  CHECK_DEVICE(x);
  return sum_impl(x, dim);
}

Tensor Device::broadcast(const Tensor &x, unsigned dim) {
  if (x.shape()[dim] != 1) {
    THROW_ERROR(
        "Cannot broadcast dimension whose value is not 1. x.shape: "
        << x.shape().to_string() << ", dim: " << dim);
  }

  CHECK_DEVICE(x);
  return broadcast_impl(x, dim);
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

}  // namespace primitiv
