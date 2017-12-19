#include <config.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

#include <Eigen/Eigen>

#include <primitiv/eigen_device.h>
#include <primitiv/error.h>

using std::cerr;
using std::endl;

template<typename T>
using EMap = ::Eigen::Map<T>;

using EArrayXf = ::Eigen::ArrayXf;
using EMatrixXf = ::Eigen::MatrixXf;

namespace primitiv {
namespace devices {

void Eigen::dump_description() const {
  cerr << "Device " << this << endl;
  cerr << "  Type: Eigen" << endl;
}

std::shared_ptr<void> Eigen::new_handle(const Shape &shape) {
  const std::uint32_t mem_size = sizeof(float) * shape.size();
  void *data = std::malloc(mem_size);
  if (!data) {
    THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  return std::shared_ptr<void>(data, std::free);
}

#define CDATA(x) static_cast<const float *>(get_handle(x))
#define MDATA(x) static_cast<float *>(get_mutable_handle(x))

#define REPEAT_OP(i, n, op) \
  for (std::uint32_t (i) = 0; (i) < (n); ++(i)) { (op); }

std::vector<float> Eigen::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  std::memcpy(&ret[0], CDATA(x), sizeof(float) * num_elements);
  return ret;
}

std::vector<std::uint32_t> Eigen::argmax_impl(const Tensor &x, std::uint32_t dim) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t repeat = s.size() / n;
  const std::uint32_t skip1 = s.lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  const float *src = CDATA(x);
  std::vector<std::uint32_t> ret;
  ret.reserve(repeat);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float max_val = src[offset];
    std::uint32_t argmax_val = 0;
    for (std::uint32_t j = 1; j < n; ++j) {
      offset += skip1;
      if (src[offset] > max_val) {
        max_val = src[offset];
        argmax_val = j;
      }
    }
    ret.emplace_back(argmax_val);
  }
  return ret;
}

std::vector<std::uint32_t> Eigen::argmin_impl(const Tensor &x, std::uint32_t dim) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t repeat = s.size() / n;
  const std::uint32_t skip1 = s.lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  const float *src = CDATA(x);
  std::vector<std::uint32_t> ret;
  ret.reserve(repeat);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float max_val = src[offset];
    std::uint32_t argmax_val = 0;
    for (std::uint32_t j = 1; j < n; ++j) {
      offset += skip1;
      if (src[offset] < max_val) {
        max_val = src[offset];
        argmax_val = j;
      }
    }
    ret.emplace_back(argmax_val);
  }
  return ret;
}

void Eigen::reset_tensor_impl(float k, Tensor &x) {
  EMap<EArrayXf>(MDATA(x), x.shape().size()).setConstant(k);
}

void Eigen::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::size_t size = x.shape().size();
  EMap<EArrayXf>(MDATA(x), size) = EMap<const EArrayXf>(values, size);
}

void Eigen::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case Device::DeviceType::NAIVE:
      reset_tensor_by_array(CDATA(x), y);
      break;
    case Device::DeviceType::EIGEN:
      reset_tensor_by_array(CDATA(x), y);
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void Eigen::identity_impl(Tensor &y) {
  const std::size_t size = y.shape()[0];
  EMap<EMatrixXf>(MDATA(y), size, size).setIdentity();
}

void Eigen::random_bernoulli_impl(float p, Tensor &y) {
  randomizer_.fill_bernoulli(p, y.shape().size(), MDATA(y));
}

void Eigen::random_uniform_impl(float lower, float upper, Tensor &y) {
  randomizer_.fill_uniform(lower, upper, y.shape().size(), MDATA(y));
}

void Eigen::random_normal_impl(float mean, float sd, Tensor &y) {
  randomizer_.fill_normal(mean, sd, y.shape().size(), MDATA(y));
}

void Eigen::random_log_normal_impl(float mean, float sd, Tensor &y) {
  randomizer_.fill_log_normal(mean, sd, y.shape().size(), MDATA(y));
}

void Eigen::pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim,
    Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t bs = y.shape().batch();
  const std::uint32_t skip_x = x.shape().has_batch() * x.shape().volume();
  const std::uint32_t skip_i = ids.size() > 1;
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / base;

  float *dest = MDATA(y);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    const float *src = CDATA(x) + batch * skip_x + base * ids[batch * skip_i];
    for (std::uint32_t i = 0; i < repeat; ++i) {
      const float *sp = src;
      REPEAT_OP(j, base, *dest++ = *sp++);
      src += skip;
    }
  }
}

void Eigen::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t repeat = y.shape().size() / span;

  float *dest = MDATA(y);
  const float *src = CDATA(x) + base * offset;
  for (std::uint32_t i = 0; i < repeat; ++i) {
    const float *sp = src;
    REPEAT_OP(j, span, *dest++ = *sp++);
    src += skip;
  }
}

void Eigen::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / skip;

  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t src_dim = x->shape()[dim];
    const std::uint32_t span = base * src_dim;
    const std::uint32_t b_skip = x->shape().has_batch() * span * repeat;
    float *dest = MDATA(y) + offset;
    const float *src = CDATA(*x);
    for (std::uint32_t batch = 0; batch < new_bs; ++batch) {
      const float *sp = src;
      for (std::uint32_t i = 0; i < repeat; ++i) {
        float *dp = dest;
        REPEAT_OP(j, span, *dp++ = *sp++);
        dest += skip;
      }
      src += b_skip;
    }
    offset += span;
  }
}

void Eigen::pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t>& ids, std::uint32_t dim,
    Tensor &gx) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_x = gx.shape().has_batch() * gx.shape().volume();
  const std::uint32_t skip_i = ids.size() > 1;
  const std::uint32_t base = gy.shape().lower_volume(dim);
  const std::uint32_t skip = base * gx.shape()[dim];
  const std::uint32_t repeat = gy.shape().volume() / base;
  const float *src = CDATA(gy);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    float *dest = MDATA(gx) + batch * skip_x + base * ids[batch * skip_i];
    for (std::uint32_t i = 0; i < repeat; ++i) {
      float *dp = dest;
      REPEAT_OP(j, base, *dp++ += *src++);
      dest += skip;
    }
  }
}

void Eigen::slice_bw_impl(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const Shape &sy = gy.shape();
  const Shape &sx = gx.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  const std::uint32_t span = base * sy[dim];
  const std::uint32_t skip = base * sx[dim];
  const std::uint32_t repeat = sx.volume() / skip;
  const std::uint32_t bs = std::max(sx.batch(), sy.batch());
  const std::uint32_t b_skip_d = sx.has_batch() * sx.volume();
  const std::uint32_t b_skip_s = sy.has_batch() * sy.volume();
  float *dest = MDATA(gx) + base * offset;
  const float *src = CDATA(gy);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    float *dp = dest;
    const float *sp = src;
    for (std::uint32_t i = 0; i < repeat; ++i) {
      float *ddp = dp;
      REPEAT_OP(j, span, *ddp++ += *sp++);
      dp += skip;
    }
    dest += b_skip_d;
    src += b_skip_s;
  }
}

#define MAYBE_USED(x) static_cast<void>(x)

#define CPUDEV_FW_X(name, op) \
void Eigen::name##_fw_impl(const Tensor &x_, Tensor &y_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); \
  EMap<EArrayXf>(MDATA(y_), size) = (op); \
}

#define CPUDEV_BW_X(name, op) \
void Eigen::name##_bw_impl( \
    const Tensor &x_, const Tensor &y_, const Tensor &gy_, Tensor &gx_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); MAYBE_USED(x); \
  EMap<const EArrayXf> y(CDATA(y_), size); MAYBE_USED(y); \
  EMap<const EArrayXf> gy(CDATA(gy_), size); \
  EMap<EArrayXf>(MDATA(gx_), size) += (op); \
}

#define CPUDEV_FW_X_CONST(name, op) \
void Eigen::name##_fw_impl(const Tensor &x_, float k, Tensor &y_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); \
  EMap<EArrayXf>(MDATA(y_), size) = (op); \
}

#define CPUDEV_BW_X_CONST(name, op) \
void Eigen::name##_bw_impl( \
    const Tensor &x_, const Tensor &y_, const Tensor &gy_, float k, \
    Tensor &gx_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); MAYBE_USED(x); \
  EMap<const EArrayXf> y(CDATA(y_), size); MAYBE_USED(y); \
  EMap<const EArrayXf> gy(CDATA(gy_), size); \
  EMap<EArrayXf>(MDATA(gx_), size) += (op); \
}

#define CPUDEV_FW_X_SCALAR(name, op) \
void Eigen::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t bs = y.shape().batch(); \
  const std::uint32_t skip_x = x.shape().has_batch() * size; \
  const std::uint32_t skip_k = k.shape().has_batch(); \
  float *dest = MDATA(y); \
  const float *src_x = CDATA(x); \
  const float *src_k = CDATA(k); \
  for (std::uint32_t batch = 0; batch < bs; ++batch) { \
    REPEAT_OP(i, size, dest[i] = (op)); \
    dest += size; \
    src_x += skip_x; \
    src_k += skip_k; \
  } \
}

#define CPUDEV_FW_AB(name, op) \
void Eigen::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t bs = y.shape().batch(); \
  const std::uint32_t skip_a = a.shape().has_batch() * size; \
  const std::uint32_t skip_b = b.shape().has_batch() * size; \
  float *dest = MDATA(y); \
  const float *src_a = CDATA(a); \
  const float *src_b = CDATA(b); \
  for (std::uint32_t batch = 0; batch < bs; ++batch) { \
    REPEAT_OP(i, size, dest[i] = (op)); \
    dest += size; \
    src_a += skip_a; \
    src_b += skip_b; \
  } \
}

CPUDEV_FW_X(negate, -x);
CPUDEV_FW_X(sqrt, x.sqrt());
CPUDEV_FW_X(exp, x.exp());
CPUDEV_FW_X(log, x.log());
CPUDEV_FW_X(tanh, x.tanh());
CPUDEV_FW_X(sigmoid, .5 + .5 * (.5 * x).tanh());
CPUDEV_FW_X(
    softplus, (x > 0.).select(
      x + (1. + (-x).exp()).log(),
      (1. + x.exp()).log()));
CPUDEV_FW_X(sin, x.sin());
CPUDEV_FW_X(cos, x.cos());
CPUDEV_FW_X(tan, x.tan());

CPUDEV_BW_X(sqrt, .5 * gy / y);
CPUDEV_BW_X(exp, gy * y);
CPUDEV_BW_X(log, gy / x);
CPUDEV_BW_X(tanh, gy * (1. - y * y));
CPUDEV_BW_X(sigmoid, gy * y * (1. - y));
CPUDEV_BW_X(softplus, gy * (.5 + .5 * (.5 * x).tanh()));
CPUDEV_BW_X(sin, gy * x.cos());
CPUDEV_BW_X(cos, -gy * x.sin());
CPUDEV_BW_X(tan, gy * (1. + y * y));

CPUDEV_FW_X_CONST(add_const, x + k);
CPUDEV_FW_X_CONST(subtract_const_r, x - k);
CPUDEV_FW_X_CONST(subtract_const_l, k - x);
CPUDEV_FW_X_CONST(multiply_const, x * k);
CPUDEV_FW_X_CONST(divide_const_r, x / k);
CPUDEV_FW_X_CONST(divide_const_l, k / x);
CPUDEV_FW_X_CONST(prelu, (x > 0.).select(x, k * x));
CPUDEV_FW_X_CONST(elu, (x > 0.).select(x, k * (x.exp() - 1.)));

CPUDEV_BW_X_CONST(add_const, gy);
CPUDEV_BW_X_CONST(subtract_const_r, gy);
CPUDEV_BW_X_CONST(subtract_const_l, -gy);
CPUDEV_BW_X_CONST(multiply_const, k * gy);
CPUDEV_BW_X_CONST(divide_const_r, gy / k);
CPUDEV_BW_X_CONST(divide_const_l, -gy * y / x);
CPUDEV_BW_X_CONST(prelu, (x > 0.).select(gy, k * gy));
CPUDEV_BW_X_CONST(elu, (x > 0.).select(gy, (y + k) * gy));

CPUDEV_FW_X_SCALAR(add_scalar, src_x[i] + *src_k);
CPUDEV_FW_X_SCALAR(subtract_scalar_r, src_x[i] - *src_k);
CPUDEV_FW_X_SCALAR(subtract_scalar_l, *src_k - src_x[i]);
CPUDEV_FW_X_SCALAR(multiply_scalar, src_x[i] * *src_k);
CPUDEV_FW_X_SCALAR(divide_scalar_r, src_x[i] / *src_k);
CPUDEV_FW_X_SCALAR(divide_scalar_l, *src_k / src_x[i]);

CPUDEV_FW_AB(add, src_a[i] + src_b[i]);
CPUDEV_FW_AB(subtract, src_a[i] - src_b[i]);
CPUDEV_FW_AB(multiply, src_a[i] * src_b[i]);
CPUDEV_FW_AB(divide, src_a[i] / src_b[i]);

#undef CPUDEV_FW_X
#undef CPUDEV_BW_X
#undef CPUDEV_FW_X_CONST
#undef CPUDEV_BW_X_CONST
#undef CPUDEV_FW_X_SCALAR
#undef CPUDEV_FW_AB

#undef MAYBE_USED

void Eigen::add_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float k = pgy[i];
      pga[i] += k;
      pgb[i] += k;
    }
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

void Eigen::subtract_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float k = pgy[i];
      pga[i] += k;
      pgb[i] -= k;
    }
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

void Eigen::multiply_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pa = CDATA(a);
  const float *pb = CDATA(b);
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float k = pgy[i];
      pga[i] += k * pb[i];
      pgb[i] += k * pa[i];
    }
    pa += skip_a;
    pb += skip_b;
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

void Eigen::divide_bw_impl(
    const Tensor &, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pb = CDATA(b);
  const float *py = CDATA(y);
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float k = pgy[i] / pb[i];
      pga[i] += k;
      pgb[i] -= k * py[i];
    }
    pb += skip_b;
    py += size;
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

void Eigen::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t di = x.shape()[0];
  const std::uint32_t dj = x.shape()[1];
  const std::uint32_t ms = di * dj;
  const std::uint32_t bs = x.shape().batch();

  const float *src = CDATA(x);
  float *dest = MDATA(y);

  for (std::uint32_t n = 0; n < bs; ++n) {
    EMap<const EMatrixXf> xx(src + n * ms, di, dj);
    EMap<EMatrixXf> yy(dest + n * ms, dj, di);
    yy.noalias() = xx.transpose();
  }
}

void Eigen::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];

  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  float *dest = MDATA(y);

  if (a.shape().has_batch()) {
    // Do multiplication multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> aa(src_a + n * a_skip, di, dj);
      EMap<const EMatrixXf> bb(src_b + n * b_skip, dj, dk);
      EMap<EMatrixXf> yy(dest + n * y_skip, di, dk);
      yy.noalias() = aa * bb;
    }
  } else {
    // Do multiplication only once using a combined matrix.
    const std::uint32_t dk_batch = dk * b.shape().batch();
    EMap<const EMatrixXf> aa(src_a, di, dj);
    EMap<const EMatrixXf> bb(src_b, dj, dk_batch);
    EMap<EMatrixXf> yy(dest, di, dk_batch);
    yy.noalias() = aa * bb;
  }
}

void Eigen::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const std::uint32_t di = gx.shape()[0];
  const std::uint32_t dj = gx.shape()[1];
  const std::uint32_t ms = di * dj;
  const std::uint32_t bs = gx.shape().batch();

  const float *src = CDATA(gy);
  float *dest = MDATA(gx);

  for (std::uint32_t n = 0; n < bs; ++n) {
    EMap<const EMatrixXf> gyy(src + n * ms, dj, di);
    EMap<EMatrixXf> gxx(dest + n * ms, di, dj);
    gxx.noalias() += gyy.transpose();
  }
}

void Eigen::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];

  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);
  const float *src_gy = CDATA(gy);
  float *dest_ga = MDATA(ga);
  float *dest_gb = MDATA(gb);

  if (a.shape().has_batch()) {
    // Do multiplication multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> aa(src_a + n * a_skip, di, dj);
      EMap<const EMatrixXf> bb(src_b + n * b_skip, dj, dk);
      EMap<const EMatrixXf> gyy(src_gy + n * y_skip, di, dk);
      EMap<EMatrixXf> gaa(dest_ga + n * a_skip, di, dj);
      EMap<EMatrixXf> gbb(dest_gb + n * b_skip, dj, dk);
      gaa.noalias() += gyy * bb.transpose();
      gbb.noalias() += aa.transpose() * gyy;
    }
  } else {
    // Do multiplication only once using a combined matrix.
    const std::uint32_t dk_batch = dk * b.shape().batch();
    EMap<const EMatrixXf> aa(src_a, di, dj);
    EMap<const EMatrixXf> bb(src_b, dj, dk_batch);
    EMap<const EMatrixXf> gyy(src_gy, di, dk_batch);
    EMap<EMatrixXf> gaa(dest_ga, di, dj);
    EMap<EMatrixXf> gbb(dest_gb, dj, dk_batch);
    gaa.noalias() += gyy * bb.transpose();
    gbb.noalias() += aa.transpose() * gyy;
  }
}

void Eigen::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t repeat = y.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float tmp = 0;
    for (std::uint32_t j = 0; j < n; ++j) {
      tmp += src[offset];
      offset += skip1;
    }
    dest[i] = tmp;
  }
}

void Eigen::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t repeat = y.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    // TODO(odashi): This calculation might generate large errors.
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float tmp = src[offset];
    for (std::uint32_t j = 1; j < n; ++j) {
      offset += skip1;
      float arg = src[offset];
      tmp = tmp > arg
        ? tmp + std::log(1. + std::exp(arg - tmp))
        : arg + std::log(1. + std::exp(tmp - arg));
    }
    dest[i] = tmp;
  }
}

void Eigen::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  const std::uint32_t repeat = x.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float tmp = src[i];
    for (std::uint32_t j = 0; j < size; ++j) {
      dest[offset] = tmp;
      offset += skip1;
    }
  }
}

void Eigen::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t size = y.shape().size();
  for (std::uint32_t i = 0; i < size; ++i) {
    float temp = 0;
    for (std::uint32_t batch = 0, pos = i; batch < bs; ++batch, pos += size) {
      temp += src[pos];
    }
    dest[i] = temp;
  }
}

void Eigen::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  float *dest = MDATA(x);
  REPEAT_OP(i, size, dest[i] *= k);
}

void Eigen::inplace_add_impl(const Tensor &x, Tensor &y) {
  const Shape &sx = x.shape();
  const Shape &sy = y.shape();
  const std::uint32_t size = sy.volume();
  const std::uint32_t bs = std::max(sx.batch(), sy.batch());
  const std::uint32_t b_skip_d = sy.has_batch() * size;
  const std::uint32_t b_skip_s = sx.has_batch() * size;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] += src[i]);
    dest += b_skip_d;
    src += b_skip_s;
  }
}

void Eigen::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  const Shape &sx = x.shape();
  const Shape &sy = y.shape();
  const std::uint32_t size = sy.volume();
  const std::uint32_t bs = std::max(sx.batch(), sy.batch());
  const std::uint32_t b_skip_d = sy.has_batch() * size;
  const std::uint32_t b_skip_s = sx.has_batch() * size;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] -= src[i]);
    dest += b_skip_d;
    src += b_skip_s;
  }
}

}  // namespace devices
}  // namespace primitiv
