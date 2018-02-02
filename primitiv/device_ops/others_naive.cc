#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

#define CPUDEV_FW_X(name, op) \
void Naive::name##_fw_impl(const Tensor &x, Tensor &y) { \
  float *dest = MDATA(y); \
  const float *src = CDATA(x); \
  const std::uint32_t size = x.shape().size(); \
  REPEAT_OP(i, size, dest[i] = (op)); \
}

#define CPUDEV_BW_X(name, op) \
void Naive::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const float *px = CDATA(x); static_cast<void>(px); \
  const float *py = CDATA(y); static_cast<void>(py); \
  const float *pgy = CDATA(gy); \
  float *pgx = MDATA(gx); \
  const std::uint32_t size = x.shape().size(); \
  REPEAT_OP(i, size, pgx[i] += (op)); \
}

#define CPUDEV_FW_X_CONST(name, op) \
void Naive::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  float *dest = MDATA(y); \
  const float *src = CDATA(x); \
  const std::uint32_t size = x.shape().size(); \
  REPEAT_OP(i, size, dest[i] = (op)); \
}

#define CPUDEV_BW_X_CONST(name, op) \
void Naive::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const float *px = CDATA(x); static_cast<void>(px); \
  const float *py = CDATA(y); static_cast<void>(py); \
  const float *pgy = CDATA(gy); \
  float *pgx = MDATA(gx); \
  const std::uint32_t size = x.shape().size(); \
  REPEAT_OP(i, size, pgx[i] += (op)); \
}

#define CPUDEV_FW_X_SCALAR(name, op) \
void Naive::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
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
void Naive::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
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

CPUDEV_FW_X(negate, -src[i]);
CPUDEV_FW_X(sqrt, std::sqrt(src[i]));
CPUDEV_FW_X(exp, std::exp(src[i]));
CPUDEV_FW_X(log, std::log(src[i]));
CPUDEV_FW_X(tanh, std::tanh(src[i]));
CPUDEV_FW_X(sigmoid, .5 + .5 * std::tanh(.5 * src[i]));
CPUDEV_FW_X(
    softplus, src[i] > 0
      ? src[i] + std::log(1 + std::exp(-src[i]))
      : std::log(1 + std::exp(src[i])));
CPUDEV_FW_X(sin, std::sin(src[i]));
CPUDEV_FW_X(cos, std::cos(src[i]));
CPUDEV_FW_X(tan, std::tan(src[i]));

CPUDEV_BW_X(sqrt, .5 * pgy[i] / py[i]);
CPUDEV_BW_X(exp, py[i] * pgy[i]);
CPUDEV_BW_X(log, pgy[i] / px[i]);
CPUDEV_BW_X(tanh, (1. - py[i] * py[i]) * pgy[i]);
CPUDEV_BW_X(sigmoid, py[i] * (1. - py[i]) * pgy[i]);
CPUDEV_BW_X(softplus, (.5 + .5 * std::tanh(.5 * px[i])) * pgy[i]);
CPUDEV_BW_X(sin, std::cos(px[i]) * pgy[i]);
CPUDEV_BW_X(cos, -std::sin(px[i]) * pgy[i]);
CPUDEV_BW_X(tan, (1 + py[i] * py[i]) * pgy[i]);

CPUDEV_FW_X_CONST(add_const, src[i] + k);
CPUDEV_FW_X_CONST(subtract_const_r, src[i] - k);
CPUDEV_FW_X_CONST(subtract_const_l, k - src[i]);
CPUDEV_FW_X_CONST(multiply_const, src[i] * k);
CPUDEV_FW_X_CONST(divide_const_r, src[i] / k);
CPUDEV_FW_X_CONST(divide_const_l, k / src[i]);
CPUDEV_FW_X_CONST(pow_const_r, std::pow(src[i], k));
CPUDEV_FW_X_CONST(pow_const_l, std::pow(k, src[i]));
CPUDEV_FW_X_CONST(prelu, src[i] * ((src[i] > 0) + k * (src[i] <= 0)));
CPUDEV_FW_X_CONST(
    elu, src[i] * (src[i] > 0) + k * (std::exp(src[i] * (src[i] <= 0)) - 1));

CPUDEV_BW_X_CONST(add_const, pgy[i]);
CPUDEV_BW_X_CONST(subtract_const_r, pgy[i]);
CPUDEV_BW_X_CONST(subtract_const_l, -pgy[i]);
CPUDEV_BW_X_CONST(multiply_const, k * pgy[i]);
CPUDEV_BW_X_CONST(divide_const_r, pgy[i] / k);
CPUDEV_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i]);
CPUDEV_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i]);
CPUDEV_BW_X_CONST(pow_const_l, std::log(k) * pgy[i] * py[i]);
CPUDEV_BW_X_CONST(prelu, pgy[i] * ((px[i] > 0) + k * (px[i] <= 0)));
CPUDEV_BW_X_CONST(elu, pgy[i] * ((px[i] > 0) + (py[i] + k) * (px[i] <= 0)));

CPUDEV_FW_X_SCALAR(add_scalar, src_x[i] + *src_k);
CPUDEV_FW_X_SCALAR(subtract_scalar_r, src_x[i] - *src_k);
CPUDEV_FW_X_SCALAR(subtract_scalar_l, *src_k - src_x[i]);
CPUDEV_FW_X_SCALAR(multiply_scalar, src_x[i] * *src_k);
CPUDEV_FW_X_SCALAR(divide_scalar_r, src_x[i] / *src_k);
CPUDEV_FW_X_SCALAR(divide_scalar_l, *src_k / src_x[i]);
CPUDEV_FW_X_SCALAR(pow_scalar_r, std::pow(src_x[i], *src_k));
CPUDEV_FW_X_SCALAR(pow_scalar_l, std::pow(*src_k, src_x[i]));

CPUDEV_FW_AB(add, src_a[i] + src_b[i]);
CPUDEV_FW_AB(subtract, src_a[i] - src_b[i]);
CPUDEV_FW_AB(multiply, src_a[i] * src_b[i]);
CPUDEV_FW_AB(divide, src_a[i] / src_b[i]);
CPUDEV_FW_AB(pow, std::pow(src_a[i], src_b[i]));

#undef CPUDEV_FW_X
#undef CPUDEV_BW_X
#undef CPUDEV_FW_X_CONST
#undef CPUDEV_BW_X_CONST
#undef CPUDEV_FW_X_SCALAR
#undef CPUDEV_FW_AB

void Naive::add_bw_impl(
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

void Naive::subtract_bw_impl(
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

void Naive::multiply_bw_impl(
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

void Naive::divide_bw_impl(
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

void Naive::pow_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pa = CDATA(a);
  const float *pb = CDATA(b);
  const float *py = CDATA(y);
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float a = pgy[i] * py[i];
      pga[i] += a * pb[i] / pa[i];
      pgb[i] += a * std::log(pa[i]);
    }
    pa += skip_a;
    pb += skip_b;
    py += size;
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

}  // namespace devices
}  // namespace primitiv
