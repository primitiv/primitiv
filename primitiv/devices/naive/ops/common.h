#ifndef PRIMITIV_DEVICES_NAIVE_OPS_COMMON_H_
#define PRIMITIV_DEVICES_NAIVE_OPS_COMMON_H_

#define MAYBE_USED(x) static_cast<void>(x)

#define CDATA(x) static_cast<const float *>(get_handle(x))
#define MDATA(x) static_cast<float *>(get_mutable_handle(x))

#define REPEAT_OP(i, n, op) \
  for (std::uint32_t i = 0; i < (n); ++i) { (op); }

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
  const float *px = CDATA(x); MAYBE_USED(px); \
  const float *py = CDATA(y); MAYBE_USED(py); \
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
  MAYBE_USED(k); \
  const float *px = CDATA(x); MAYBE_USED(px); \
  const float *py = CDATA(y); MAYBE_USED(py); \
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

#endif  // PRIMITIV_DEVICES_NAIVE_OPS_COMMON_H_
