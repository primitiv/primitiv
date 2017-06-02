#include <config.h>

#include <primitiv/device.h>
#include <primitiv/error.h>

// NOTE(odashi): This source only checks shape prerequisites of each operation.

#define CHECK_DEVICE(x) \
  if ((x).device() != this) { \
    THROW_ERROR( \
        "Device mismatched. (" #x ").device(): " << (x).device() \
        << "!= this: " << this); \
  }

namespace primitiv {

Tensor Device::new_tensor(const Shape &shape) {
  return new_tensor_impl(shape);
}

Tensor Device::new_tensor(const Shape &shape, float k) {
  Tensor ret = new_tensor_impl(shape);
  reset_tensor_impl(ret, k);
  return ret;
}

Tensor Device::new_tensor(
    const Shape &shape, const std::vector<float> &values) {
  Tensor ret = new_tensor_impl(shape);
  reset_tensor_impl(ret, values);
  return ret;
}

void Device::delete_tensor(Tensor &x) {
  delete_tensor_impl(x);
}

std::vector<float> Device::tensor_to_vector(const Tensor &x) {
  CHECK_DEVICE(x);
  return tensor_to_vector_impl(x);
}

void Device::reset_tensor(Tensor &x, float k) {
  CHECK_DEVICE(x);
  reset_tensor_impl(x, k);
}

void Device::reset_tensor(Tensor &x, const std::vector<float> &values) {
  CHECK_DEVICE(x);
  const unsigned num_elements = x.shape().size();
  if (values.size() != num_elements) {
    THROW_ERROR(
        "Data sizes mismatched. required: " << num_elements
        << " (shape: " << x.shape().to_string() << ") != actual: "
        << values.size());
  }
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
  CHECK_DEVICE(x);
  const Shape &s = x.shape();
  if (lower >= upper || upper > s.dim(dim)) {
    THROW_ERROR(
        "Attempted to invalid slicing. x.shape: " << s.to_string()
        << ", dim: " << dim << ", lower: " << lower << ", upper: " << upper);
  }

  if (dim >= s.dims().size()) {
    // Resulting tensor is completely same as the argument.
    return duplicate(x);
  }

  return slice_impl(x, dim, lower, upper);
}

Tensor Device::concat(const std::vector<const Tensor *> &xs, unsigned dim) {
  if (xs.empty()) {
    THROW_ERROR("No tensors to be concatenated.");
  }

  CHECK_DEVICE(*xs[0]);
  std::vector<unsigned> ref_dims = xs[0]->shape().dims();
  unsigned sum_dim = 1;
  if (dim < ref_dims.size()) { sum_dim = ref_dims[dim]; ref_dims[dim] = 1; }
  while (!ref_dims.empty() && ref_dims.back() == 1) ref_dims.pop_back();
  unsigned ref_bs = xs[0]->shape().batch_size();

  bool ok = true;
  for (unsigned i = 1; i < xs.size(); ++i) {
    CHECK_DEVICE(*xs[i]);
    std::vector<unsigned> dims = xs[i]->shape().dims();
    if (dim < dims.size()) { sum_dim += dims[dim]; dims[dim] = 1; }
    else ++sum_dim;
    while (!dims.empty() && dims.back() == 1) dims.pop_back();
    if (dims != ref_dims) { ok = false; break; }
    unsigned bs = xs[1]->shape().batch_size();
    if (bs != ref_bs && bs > 1 && ref_bs > 1) { ok = false; break; }
    if (ref_bs == 1) ref_bs = bs;
  }

  if (!ok) {
    std::string dims_str = xs[0]->shape().to_string();
    for (unsigned i = 1; i < xs.size(); ++i) {
      dims_str += ", " + xs[i]->shape().to_string();
    }
    THROW_ERROR("Attempted to concatenate tensors with shapes: " << dims_str);
  }

  if (dim >= ref_dims.size()) {
    ref_dims.insert(ref_dims.end(), dim - ref_dims.size() + 1, 1);
  }
  ref_dims[dim] = sum_dim;

  return concat_impl(xs, dim, Shape(ref_dims, ref_bs));
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
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) {
    THROW_ERROR(
        "Attempted to add tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }
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
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) {
    THROW_ERROR(
        "Attempted to subtract tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }
  return subtract_impl(a, b);
}

Tensor Device::multiply(const Tensor &x, float k) {
  CHECK_DEVICE(x);
  return multiply_impl(x, k);
}

Tensor Device::multiply(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) {
    THROW_ERROR(
        "Attempted to multiply tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }
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
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) {
    THROW_ERROR(
        "Attempted to divide tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }
  return divide_impl(a, b);
}

Tensor Device::transpose(const Tensor &x) {
  CHECK_DEVICE(x);
  const Shape &s = x.shape();
  if (s.dims().size() > 2) {
    THROW_ERROR(
        "Attempted to transpose a tensor with shape " << s.to_string() << '.');
  }
  return transpose_impl(x);
}

Tensor Device::dot(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  if (sa.dims().size() > 2 || sb.dims().size() > 2 ||
      sa.dim(1) != sb.dim(0) ||
      (ba != bb && ba > 1 && bb > 1)) {
    THROW_ERROR(
        "Attempted to calculate the dot product of tensors with shapes "
        << sa.to_string() << " and " << sb.to_string() << '.');
  }
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

Tensor Device::batch_sum(const Tensor &x) {
  CHECK_DEVICE(x);
  return batch_sum_impl(x);
}

void Device::add_gradient(Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) {
    THROW_ERROR(
        "Attempted to add gradients with shape "
        << sb.to_string() << " to " << sa.to_string() << '.');
  }
  add_gradient_impl(a, b);
}

}  // namespace primitiv
