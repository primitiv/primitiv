#include <primitiv/config.h>

#include <primitiv/device.h>
#include <primitiv/error.h>
#include <primitiv/shape_ops.h>

using std::vector;

// NOTE(odashi): This source only checks shape prerequisites of each operation.

#define CHECK_DEVICE(x) \
  if (&(x).device() != this) { \
    THROW_ERROR( \
        "Device mismatched. &(" #x ").device(): " << &(x).device() \
        << " != this: " << this); \
  }

namespace primitiv {

Tensor Device::new_raw_tensor(const Shape &shape) {
  return Tensor(shape, *this, new_handle(shape));
}

Tensor Device::new_tensor_by_constant(const Shape &shape, float k) {
  Tensor ret(shape, *this, new_handle(shape));
  reset_tensor(k, ret);
  return ret;
}

Tensor Device::new_tensor_by_array(const Shape &shape, const float values[]) {
  Tensor ret(shape, *this, new_handle(shape));
  reset_tensor_by_array(values, ret);
  return ret;
}

Tensor Device::new_tensor_by_vector(
    const Shape &shape, const vector<float> &values) {
  Tensor ret(shape, *this, new_handle(shape));
  reset_tensor_by_vector(values, ret);
  return ret;
}

vector<float> Device::tensor_to_vector(const Tensor &x) {
  CHECK_DEVICE(x);
  return tensor_to_vector_impl(x);
}

vector<std::uint32_t> Device::argmax(const Tensor &x, std::uint32_t dim) {
  CHECK_DEVICE(x);
  return argmax_impl(x, dim);
}

vector<std::uint32_t> Device::argmin(const Tensor &x, std::uint32_t dim) {
  CHECK_DEVICE(x);
  return argmin_impl(x, dim);
}

void Device::reset_tensor(float k, Tensor &x) {
  CHECK_DEVICE(x);
  reset_tensor_impl(k, x);
}

void Device::reset_tensor_by_array(const float values[], Tensor &x) {
  // NOTE(odashi):
  // There is no method to guarantee the size of the array for now.
  CHECK_DEVICE(x);
  reset_tensor_by_array_impl(values, x);
}

void Device::reset_tensor_by_vector(const vector<float> &values, Tensor &x) {
  CHECK_DEVICE(x);
  if (values.size() != x.shape().size()) {
    THROW_ERROR(
        "Data sizes mismatched. required: " << x.shape().size()
        << " (shape: " << x.shape().to_string() << ") != actual: "
        << values.size());
  }
  reset_tensor_by_array_impl(values.data(), x);
}

Tensor Device::copy_tensor(const Tensor &x) {
  // NOTE(odashi):
  // This function should return always different memory with x.
  if (!x.valid()) THROW_ERROR("Attempted to copy an invalid tensor.");
  Tensor y = new_raw_tensor(x.shape());
  copy_tensor_impl(x, y);
  return y;
}

Tensor Device::identity(std::uint32_t size) {
  if (size == 0) {
    THROW_ERROR("Invalid size of the identity matrix: " << size);
  }
  Tensor y = new_raw_tensor({size, size});
  identity_impl(y);
  return y;
}

Tensor Device::random_bernoulli(const Shape &shape, float p) {
  if (p < 0 || p > 1) {
    THROW_ERROR("Invalid Bernoulli probability: " << p);
  }
  Tensor y = new_raw_tensor(shape);
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
  Tensor y = new_raw_tensor(shape);
  random_uniform_impl(lower, upper, y);
  return y;
}

Tensor Device::random_normal(const Shape &shape, float mean, float sd) {
  if (sd <= 0) {
    THROW_ERROR(
        "Invalid parameter of the normal distribution. mean: " << mean
        << ", SD: " << sd);
  }
  Tensor y = new_raw_tensor(shape);
  random_normal_impl(mean, sd, y);
  return y;
}

Tensor Device::random_log_normal(const Shape &shape, float mean, float sd) {
  if (sd <= 0) {
    THROW_ERROR(
        "Invalid parameter of the log-normal distribution. mean: " << mean
        << ", SD: " << sd);
  }
  Tensor y = new_raw_tensor(shape);
  random_log_normal_impl(mean, sd, y);
  return y;
}

Tensor Device::pick_fw(
    const Tensor &x, const vector<std::uint32_t> &ids, std::uint32_t dim) {
  CHECK_DEVICE(x);
  Tensor y = new_raw_tensor(shape_ops::pick(x.shape(), ids, dim));
  pick_fw_impl(x, ids, dim, y);
  return y;
}

Tensor Device::slice_fw(
    const Tensor &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper) {
  CHECK_DEVICE(x);
  Tensor y = new_raw_tensor(shape_ops::slice(x.shape(), dim, lower, upper));
  slice_fw_impl(x, dim, lower, y);
  return y;
}

Tensor Device::concat_fw(const vector<const Tensor *> &xs, std::uint32_t dim) {
  if (xs.empty()) THROW_ERROR("No tensors to concat.");
  vector<Shape> shapes;
  shapes.reserve(xs.size());

  for (std::uint32_t i = 0; i < xs.size(); ++i) {
    CHECK_DEVICE(*xs[i]);
    shapes.emplace_back(xs[i]->shape());
  }

  Tensor y = new_raw_tensor(shape_ops::concat(shapes, dim));
  concat_fw_impl(xs, dim, y);
  return y;
}

void Device::pick_bw(
    const Tensor &gy, const std::vector<std::uint32_t> &ids, std::uint32_t dim,
    Tensor &gx) {
  CHECK_DEVICE(gy);
  CHECK_DEVICE(gx);
  const Shape sy = shape_ops::pick(gx.shape(), ids, dim);
  if (gy.shape() != sy) {
    THROW_ERROR(
        "Shape mismatched. gy.shape(): " << gy.shape().to_string()
        << " != expected shape: " << sy.to_string());
  }
  pick_bw_impl(gy, ids, dim, gx);
}

void Device::slice_bw(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  CHECK_DEVICE(gy);
  CHECK_DEVICE(gx);
  const Shape &sy = gy.shape();
  const Shape &sx = gx.shape();
  if (!sy.has_same_loo_dims(sx, dim) || !sy.has_compatible_batch(sx) ||
      offset + sy[dim] > sx[dim]) {
    THROW_ERROR(
        "Attempted to add gradients with shape "
        << sy.to_string() << ", dim " << dim << ", offset " << offset
        << " to shape" << sx.to_string() << '.');
  }
  if (dim >= sx.depth()) inplace_add_impl(gy, gx);
  else slice_bw_impl(gy, dim, offset, gx);
}

#define DEV_FW_X(name, sop) \
Tensor Device::name##_fw(const Tensor &x) { \
  CHECK_DEVICE(x); \
  Tensor y = new_raw_tensor(sop(x.shape())); \
  name##_fw_impl(x, y); \
  return y; \
}

#define DEV_BW_X(name, sop) \
void Device::name##_bw( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  CHECK_DEVICE(x); \
  CHECK_DEVICE(y); \
  CHECK_DEVICE(gy); \
  CHECK_DEVICE(gx); \
  if (x.shape() != gx.shape() || \
      y.shape() != gy.shape() || \
      y.shape() != sop(x.shape())) { \
    THROW_ERROR( \
        "Shape mismatched at " #name "_bw" \
        << ". x.shape: " << x.shape().to_string() \
        << ", y.shape: " << y.shape().to_string() \
        << ", gy.shape: " << gy.shape().to_string() \
        << ", gx.shape: " << gx.shape().to_string()); \
  } \
  name##_bw_impl(x, y, gy, gx); \
}

#define DEV_FW_X_CONST(name) \
Tensor Device::name##_fw(const Tensor &x, float k) { \
  CHECK_DEVICE(x); \
  Tensor y = new_raw_tensor(x.shape()); \
  name##_fw_impl(x, k, y); \
  return y; \
}

#define DEV_BW_X_CONST(name) \
void Device::name##_bw( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  CHECK_DEVICE(x); \
  CHECK_DEVICE(y); \
  CHECK_DEVICE(gy); \
  CHECK_DEVICE(gx); \
  const Shape &s = x.shape(); \
  if (y.shape() != s || gy.shape() != s || gx.shape() != s) { \
    THROW_ERROR( \
        "Shape mismatched at " #name "_bw" \
        << ". x.shape: " << s.to_string() \
        << ", y.shape: " << y.shape().to_string() \
        << ", gy.shape: " << gy.shape().to_string() \
        << ", gx.shape: " << gx.shape().to_string()); \
  } \
  name##_bw_impl(x, y, gy, k, gx); \
}

#define DEV_FW_AB(name, sop) \
Tensor Device::name##_fw(const Tensor &a, const Tensor &b) { \
  CHECK_DEVICE(a); \
  CHECK_DEVICE(b); \
  Tensor y = new_raw_tensor(sop(a.shape(), b.shape())); \
  name##_fw_impl(a, b, y); \
  return y; \
}

#define DEV_BW_AB(name, sop) \
void Device::name##_bw( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  CHECK_DEVICE(a); \
  CHECK_DEVICE(b); \
  CHECK_DEVICE(y); \
  CHECK_DEVICE(gy); \
  CHECK_DEVICE(ga); \
  CHECK_DEVICE(gb); \
  if (a.shape() != ga.shape() || \
      b.shape() != gb.shape() || \
      y.shape() != gy.shape() || \
      y.shape() != sop(a.shape(), b.shape())) { \
    THROW_ERROR( \
        "Shape mismatched at " #name "_bw" \
        << ". a.shape: " << a.shape().to_string() \
        << ", b.shape: " << b.shape().to_string() \
        << ", y.shape: " << y.shape().to_string() \
        << ", gy.shape: " << gy.shape().to_string() \
        << ", ga.shape: " << ga.shape().to_string() \
        << ", gb.shape: " << gb.shape().to_string()); \
  } \
  name##_bw_impl(a, b, y, gy, ga, gb); \
}

DEV_FW_X(negate, static_cast<const Shape &>);
DEV_FW_X(sqrt, static_cast<const Shape &>);
DEV_FW_X(exp, static_cast<const Shape &>);
DEV_FW_X(log, static_cast<const Shape &>);
DEV_FW_X(tanh, static_cast<const Shape &>);
DEV_FW_X(sigmoid, static_cast<const Shape &>);
DEV_FW_X(softplus, static_cast<const Shape &>);
DEV_FW_X(sin, static_cast<const Shape &>);
DEV_FW_X(cos, static_cast<const Shape &>);
DEV_FW_X(tan, static_cast<const Shape &>);
DEV_FW_X(transpose, shape_ops::transpose);

DEV_BW_X(sqrt, static_cast<const Shape &>);
DEV_BW_X(exp, static_cast<const Shape &>);
DEV_BW_X(log, static_cast<const Shape &>);
DEV_BW_X(tanh, static_cast<const Shape &>);
DEV_BW_X(sigmoid, static_cast<const Shape &>);
DEV_BW_X(softplus, static_cast<const Shape &>);
DEV_BW_X(sin, static_cast<const Shape &>);
DEV_BW_X(cos, static_cast<const Shape &>);
DEV_BW_X(tan, static_cast<const Shape &>);
DEV_BW_X(transpose, shape_ops::transpose);

DEV_FW_X_CONST(add_const);
DEV_FW_X_CONST(subtract_const_r);
DEV_FW_X_CONST(subtract_const_l);
DEV_FW_X_CONST(multiply_const);
DEV_FW_X_CONST(divide_const_r);
DEV_FW_X_CONST(divide_const_l);
DEV_FW_X_CONST(pow_const_r);
DEV_FW_X_CONST(pow_const_l);
DEV_FW_X_CONST(prelu);
DEV_FW_X_CONST(elu);

DEV_BW_X_CONST(add_const);
DEV_BW_X_CONST(subtract_const_r);
DEV_BW_X_CONST(subtract_const_l);
DEV_BW_X_CONST(multiply_const);
DEV_BW_X_CONST(divide_const_r);
DEV_BW_X_CONST(divide_const_l);
DEV_BW_X_CONST(pow_const_r);
DEV_BW_X_CONST(pow_const_l);
DEV_BW_X_CONST(prelu);
DEV_BW_X_CONST(elu);

DEV_FW_AB(add_scalar, shape_ops::scalar_op);
DEV_FW_AB(subtract_scalar_r, shape_ops::scalar_op);
DEV_FW_AB(subtract_scalar_l, shape_ops::scalar_op);
DEV_FW_AB(multiply_scalar, shape_ops::scalar_op);
DEV_FW_AB(divide_scalar_r, shape_ops::scalar_op);
DEV_FW_AB(divide_scalar_l, shape_ops::scalar_op);
DEV_FW_AB(pow_scalar_r, shape_ops::scalar_op);
DEV_FW_AB(pow_scalar_l, shape_ops::scalar_op);

DEV_FW_AB(add, shape_ops::elementwise);
DEV_FW_AB(subtract, shape_ops::elementwise);
DEV_FW_AB(multiply, shape_ops::elementwise);
DEV_FW_AB(divide, shape_ops::elementwise);
DEV_FW_AB(pow, shape_ops::elementwise);
DEV_FW_AB(matmul, shape_ops::matmul);

DEV_BW_AB(add, shape_ops::elementwise);
DEV_BW_AB(subtract, shape_ops::elementwise);
DEV_BW_AB(multiply, shape_ops::elementwise);
DEV_BW_AB(divide, shape_ops::elementwise);
DEV_BW_AB(pow, shape_ops::elementwise);
DEV_BW_AB(matmul, shape_ops::matmul);

#undef DEV_FW_X
#undef DEV_BW_X
#undef DEV_FW_X_CONST
#undef DEV_BW_X_CONST
#undef DEV_FW_AB
#undef DEV_BW_AB

Tensor Device::sum_fw(const Tensor &x, std::uint32_t dim) {
  CHECK_DEVICE(x);
  Tensor y = new_raw_tensor(x.shape().resize_dim(dim, 1));
  sum_fw_impl(x, dim, y);
  return y;
}

Tensor Device::logsumexp_fw(const Tensor &x, std::uint32_t dim) {
  CHECK_DEVICE(x);
  Tensor y = new_raw_tensor(x.shape().resize_dim(dim, 1));
  logsumexp_fw_impl(x, dim, y);
  return y;
}

Tensor Device::broadcast_fw(const Tensor &x, std::uint32_t dim, std::uint32_t size) {
  CHECK_DEVICE(x);
  Tensor y = new_raw_tensor(shape_ops::broadcast(x.shape(), dim, size));
  broadcast_fw_impl(x, dim, size, y);
  return y;
}

Tensor Device::batch_sum_fw(const Tensor &x) {
  CHECK_DEVICE(x);
  Tensor y = new_raw_tensor(x.shape().resize_batch(1));
  batch_sum_fw_impl(x, y);
  return y;
}

void Device::inplace_multiply_const(float k, Tensor &x) {
  CHECK_DEVICE(x);
  inplace_multiply_const_impl(k, x);
}

void Device::inplace_add(const Tensor &x, Tensor &y) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(y);
  const Shape &sx = x.shape();
  const Shape &sy = y.shape();
  if (!sx.has_same_dims(sy) || !sx.has_compatible_batch(sy)) {
    THROW_ERROR(
        "Attempted to add values of shape "
        << sx.to_string() << " to " << sy.to_string() << '.');
  }
  inplace_add_impl(x, y);
}

void Device::inplace_subtract(const Tensor &x, Tensor &y) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(y);
  const Shape &sx = x.shape();
  const Shape &sy = y.shape();
  if (!sx.has_same_dims(sy) || !sx.has_compatible_batch(sy)) {
    THROW_ERROR(
        "Attempted to subtract values of shape "
        << sx.to_string() << " from " << sy.to_string() << '.');
  }
  inplace_subtract_impl(x, y);
}

}  // namespace primitiv
