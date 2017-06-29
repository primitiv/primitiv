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
        << " != this: " << this); \
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
  CHECK_DEVICE(x);
  if (values.size() != x.shape().size()) {
    THROW_ERROR(
        "Data sizes mismatched. required: " << x.shape().size()
        << " (shape: " << x.shape().to_string() << ") != actual: "
        << values.size());
  }
  reset_tensor_by_array_impl(x, values.data());
}

Tensor Device::copy_tensor(const Tensor &x) {
  if (!x.valid()) THROW_ERROR("Attempted to copy an invalid tensor.");
  Tensor y = new_tensor(x.shape());
  copy_tensor_impl(x, y);
  return y;
}

Tensor Device::random_bernoulli(const Shape &shape, float p) {
  if (p < 0 || p > 1) {
    THROW_ERROR("Invalid Bernoulli probability: " << p);
  }
  Tensor y = new_tensor(shape);
  random_bernoulli_impl(p, y);
  return y;
}

Tensor Device::random_uniform(
    const Shape &shape, float lower, float upper) {
  if (upper < lower) {
    THROW_ERROR(
        "Invalid parameter of the uniform distribution. lower: " << lower
        << ", upper: " << upper);
  }
  Tensor y = new_tensor(shape);
  random_uniform_impl(lower, upper, y);
  return y;
}

Tensor Device::random_normal(const Shape &shape, float mean, float sd) {
  if (sd <= 0) {
    THROW_ERROR(
        "Invalid parameter of the normal distribution. mean: " << mean
        << ", SD: " << sd);
  }
  Tensor y = new_tensor(shape);
  random_normal_impl(mean, sd, y);
  return y;
}

Tensor Device::random_log_normal(const Shape &shape, float mean, float sd) {
  if (sd <= 0) {
    THROW_ERROR(
        "Invalid parameter of the log-normal distribution. mean: " << mean
        << ", SD: " << sd);
  }
  Tensor y = new_tensor(shape);
  random_log_normal_impl(mean, sd, y);
  return y;
}

Tensor Device::pick_fw(
    const Tensor &x, unsigned dim, const vector<unsigned> &ids) {
  CHECK_DEVICE(x);
  return pick_fw_impl(x, dim, ids, shape_ops::pick(x.shape(), dim, ids));
}

Tensor Device::slice_fw(
    const Tensor &x, unsigned dim, unsigned lower, unsigned upper) {
  CHECK_DEVICE(x);
  return slice_fw_impl(
      x, dim, lower, shape_ops::slice(x.shape(), dim, lower, upper));
}

Tensor Device::concat_fw(const vector<const Tensor *> &xs, unsigned dim) {
  if (xs.empty()) THROW_ERROR("No tensors to concat.");
  vector<const Shape *> shapes(xs.size());
  for (unsigned i = 0; i < xs.size(); ++i) {
    CHECK_DEVICE(*xs[i]);
    shapes[i] = &xs[i]->shape();
  }
  return concat_fw_impl(xs, dim, shape_ops::concat(shapes, dim));
}

#define DEV_FW_X(name) \
Tensor Device::name##_fw(const Tensor &x) { \
  CHECK_DEVICE(x); \
  return name##_fw_impl(x); \
}

#define DEV_FW_X_CONST(name) \
Tensor Device::name##_fw(const Tensor &x, float k) { \
  CHECK_DEVICE(x); \
  return name##_fw_impl(x, k); \
}

#define DEV_FW_X_SCALAR(name) \
Tensor Device::name##_fw(const Tensor &x, const Tensor &k) { \
  CHECK_DEVICE(x); \
  CHECK_DEVICE(k); \
  return name##_fw_impl(x, k, shape_ops::scalar_op(x.shape(), k.shape())); \
}

#define DEV_FW_AB(name) \
Tensor Device::name##_fw(const Tensor &a, const Tensor &b) { \
  CHECK_DEVICE(a); \
  CHECK_DEVICE(b); \
  return name##_fw_impl(a, b, shape_ops::elementwise(a.shape(), b.shape())); \
}

DEV_FW_X(negate);
DEV_FW_X(sqrt);
DEV_FW_X(exp);
DEV_FW_X(tanh);
DEV_FW_X(sigmoid);
DEV_FW_X(sin);
DEV_FW_X(cos);
DEV_FW_X(tan);

DEV_FW_X_CONST(add_const);
DEV_FW_X_CONST(subtract_const_r);
DEV_FW_X_CONST(subtract_const_l);
DEV_FW_X_CONST(multiply_const);
DEV_FW_X_CONST(divide_const_r);
DEV_FW_X_CONST(divide_const_l);
DEV_FW_X_CONST(pstep);
DEV_FW_X_CONST(prelu);

DEV_FW_X_SCALAR(add_scalar);
DEV_FW_X_SCALAR(subtract_scalar_r);
DEV_FW_X_SCALAR(subtract_scalar_l);
DEV_FW_X_SCALAR(multiply_scalar);
DEV_FW_X_SCALAR(divide_scalar_r);
DEV_FW_X_SCALAR(divide_scalar_l);

DEV_FW_AB(add);
DEV_FW_AB(subtract);
DEV_FW_AB(multiply);
DEV_FW_AB(divide);

#undef DEV_FW_X
#undef DEV_FW_X_CONST
#undef DEV_FW_X_SCALAR
#undef DEV_FW_AB

Tensor Device::transpose_fw(const Tensor &x) {
  CHECK_DEVICE(x);
  return transpose_fw_impl(x, shape_ops::transpose(x.shape()));
}

Tensor Device::matmul_fw(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  return matmul_fw_impl(a, b, shape_ops::matmul(a.shape(), b.shape()));
}

void Device::matmul_bw(
    const Tensor &a, const Tensor &b, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  CHECK_DEVICE(gy);
  CHECK_DEVICE(ga);
  CHECK_DEVICE(gb);
  if (a.shape() != ga.shape() || b.shape() != gb.shape() ||
      gy.shape() != shape_ops::matmul(a.shape(), b.shape())) {
    THROW_ERROR(
        "Shape mismatched at matmul_bw"
        << ". a.shape: " << a.shape().to_string()
        << ", b.shape: " << b.shape().to_string()
        << ", gy.shape: " << gy.shape().to_string()
        << ", ga.shape: " << ga.shape().to_string()
        << ", gb.shape: " << gb.shape().to_string());
  }
  matmul_bw_impl(a, b, gy, ga, gb);
}

Tensor Device::sum_fw(const Tensor &x, unsigned dim) {
  CHECK_DEVICE(x);
  return sum_fw_impl(x, dim);
}

Tensor Device::logsumexp_fw(const Tensor &x, unsigned dim) {
  CHECK_DEVICE(x);
  return logsumexp_fw_impl(x, dim);
}

Tensor Device::broadcast_fw(const Tensor &x, unsigned dim, unsigned size) {
  CHECK_DEVICE(x);
  return broadcast_fw_impl(
      x, dim, size, shape_ops::broadcast(x.shape(), dim, size));
}

Tensor Device::batch_sum_fw(const Tensor &x) {
  CHECK_DEVICE(x);
  return batch_sum_fw_impl(x);
}

void Device::add_gradient(Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_dims(sb) || !sa.has_compatible_batch(sb)) {
    THROW_ERROR(
        "Attempted to add gradients with shape "
        << sb.to_string() << " to " << sa.to_string() << '.');
  }
  add_gradient_impl(a, b);
}

void Device::add_gradient_offset(
    Tensor &a, const Tensor &b, unsigned dim, unsigned offset) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  if (!sa.has_same_loo_dims(sb, dim) || !sa.has_compatible_batch(sb) ||
      offset + sb[dim] > sa[dim]) {
    THROW_ERROR(
        "Attempted to add gradients with shape "
        << sb.to_string() << ", dim " << dim << ", offset " << offset
        << " to shape" << sa.to_string() << '.');
  }
  if (dim >= sa.depth()) add_gradient(a, b);
  else add_gradient_offset_impl(a, b, dim, offset);
}

void Device::add_gradient_sparse(
    Tensor &a, const Tensor &b,
    unsigned dim, const std::vector<unsigned> &ids) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape sb = shape_ops::pick(a.shape(), dim, ids);
  if (sb != b.shape()) {
    THROW_ERROR(
        "Shape mismatched. b.shape(): " << b.shape().to_string()
        << " != expected shape: " << sb.to_string());
  }
  add_gradient_sparse_impl(a, b, dim, ids);
}

}  // namespace primitiv
