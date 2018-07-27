#ifndef PRIMITIV_DEVICES_EIGEN_OPS_COMMON_H_
#define PRIMITIV_DEVICES_EIGEN_OPS_COMMON_H_

// NOTE(vbkaisetsu):
// Eigen contains a few LGPL-licensed features. They conflict with
// Apache License version 2.
// EIGEN_MPL2_ONLY guarantees that primitiv does not use LGPL-licensed
// features.
//
// For more ditails, see:
// http://eigen.tuxfamily.org/index.php?title=Main_Page#License
#define EIGEN_MPL2_ONLY
#include <Eigen/Eigen>

template<typename T>
using EMap = ::Eigen::Map<T>;

using EArrayXf = ::Eigen::ArrayXf;
using EMatrixXf = ::Eigen::MatrixXf;

#define CDATA(x) static_cast<const float *>(get_handle(x))
#define MDATA(x) static_cast<float *>(get_mutable_handle(x))

#define MAYBE_USED(x) static_cast<void>(x)

#define REPEAT_OP(i, n, op) \
  for (std::uint32_t i = 0; i < (n); ++i) { (op); }

#define EIGEN_DEV_FW_X(name, op) \
void Eigen::name##_fw_impl(const Tensor &x_, Tensor &y_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); \
  EMap<EArrayXf>(MDATA(y_), size) = (op); \
}

#define EIGEN_DEV_BW_X(name, op) \
void Eigen::name##_bw_impl( \
    const Tensor &x_, const Tensor &y_, const Tensor &gy_, Tensor &gx_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); MAYBE_USED(x); \
  EMap<const EArrayXf> y(CDATA(y_), size); MAYBE_USED(y); \
  EMap<const EArrayXf> gy(CDATA(gy_), size); \
  EMap<EArrayXf>(MDATA(gx_), size) += (op); \
}

#define EIGEN_DEV_FW_X_CONST(name, op) \
void Eigen::name##_fw_impl(const Tensor &x_, float k, Tensor &y_) { \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); \
  EMap<EArrayXf>(MDATA(y_), size) = (op); \
}

#define EIGEN_DEV_BW_X_CONST(name, op) \
void Eigen::name##_bw_impl( \
    const Tensor &x_, const Tensor &y_, const Tensor &gy_, float k, \
    Tensor &gx_) { \
  MAYBE_USED(k); \
  const std::size_t size = x_.shape().size(); \
  EMap<const EArrayXf> x(CDATA(x_), size); MAYBE_USED(x); \
  EMap<const EArrayXf> y(CDATA(y_), size); MAYBE_USED(y); \
  EMap<const EArrayXf> gy(CDATA(gy_), size); \
  EMap<EArrayXf>(MDATA(gx_), size) += (op); \
}

#define EIGEN_DEV_FW_X_SCALAR(name, op) \
void Eigen::name##_fw_impl(const Tensor &x_, const Tensor &k_, Tensor &y_) { \
  const std::uint32_t size = y_.shape().volume(); \
  const std::uint32_t bs = y_.shape().batch(); \
  const std::uint32_t skip_x = x_.shape().has_batch() * size; \
  const std::uint32_t skip_k = k_.shape().has_batch(); \
  const float *src_x = CDATA(x_); \
  const float *src_k = CDATA(k_); \
  float *dest = MDATA(y_); \
  for (std::uint32_t batch = 0; batch < bs; ++batch) { \
    EMap<const EArrayXf> x(src_x, size); \
    const float k = *src_k; \
    EMap<EArrayXf>(dest, size) = (op); \
    dest += size; \
    src_x += skip_x; \
    src_k += skip_k; \
  } \
}

#define EIGEN_DEV_FW_AB(name, op) \
void Eigen::name##_fw_impl(const Tensor &a_, const Tensor &b_, Tensor &y_) { \
  const std::uint32_t size = y_.shape().volume(); \
  const std::uint32_t bs = y_.shape().batch(); \
  const std::uint32_t skip_a = a_.shape().has_batch() * size; \
  const std::uint32_t skip_b = b_.shape().has_batch() * size; \
  const float *src_a = CDATA(a_); \
  const float *src_b = CDATA(b_); \
  float *dest = MDATA(y_); \
  for (std::uint32_t batch = 0; batch < bs; ++batch) { \
    EMap<const EArrayXf> a(src_a, size); \
    EMap<const EArrayXf> b(src_b, size); \
    EMap<EArrayXf>(dest, size) = (op); \
    dest += size; \
    src_a += skip_a; \
    src_b += skip_b; \
  } \
}

#endif  // PRIMITIV_DEVICES_EIGEN_OPS_COMMON_H_
