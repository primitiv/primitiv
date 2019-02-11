inline void atomic_add_float(global float *source, const float operand) {
  union {
    unsigned u;
    float f;
  } oldval, newval;
  unsigned readback;
  oldval.f = *source;
  newval.f = oldval.f + operand;
  while ((readback = atomic_cmpxchg(
      (global unsigned *) source, oldval.u, newval.u)) != oldval.u) {
    oldval.u = readback;
    newval.f = oldval.f + operand;
  }
}

#define OPENCLDEV_KERNEL_FW_X(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const unsigned size, global float *py) { \
    const unsigned i = get_global_id(0); \
  if (i < size) py[i] = (op); \
}

#define OPENCLDEV_KERNEL_BW_X(name, op) \
kernel void name##_bw_kernel( \
    const global float *px, const global float *py, const global float *pgy, \
    const unsigned size, global float *pgx) { \
   const unsigned i = get_global_id(0); \
   if (i < size) pgx[i] += (op); \
}

#define OPENCLDEV_KERNEL_FW_X_CONST(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const float k, \
    const unsigned size, global float *py) { \
  const unsigned i = get_global_id(0); \
  if (i < size) py[i] = (op); \
 }

#define OPENCLDEV_KERNEL_BW_X_CONST(name, op) \
kernel void name##_bw_kernel( \
    const global float *px, const global float *py, const global float *pgy, \
    const float k, const unsigned size, global float *pgx) { \
  const unsigned i = get_global_id(0); \
  if (i < size) pgx[i] += (op); \
}

#define OPENCLDEV_KERNEL_FW_X_SCALAR_R(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const global float *pk, const unsigned size, \
    const unsigned mbx, const unsigned mbk, global float *py) { \
  const unsigned i = get_global_id(0); \
  const unsigned bid_y = get_group_id(1); \
  const unsigned shift = bid_y * size; \
  if (i < size) { \
    py[i + shift] = op(px[i + mbx * shift], pk[mbk * bid_y]); \
  } \
}

#define OPENCLDEV_KERNEL_FW_X_SCALAR_L(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const global float *pk, const unsigned size, \
    const unsigned mbx, const unsigned mbk, global float *py) { \
  const unsigned i = get_global_id(0); \
  const unsigned bid_y = get_group_id(1); \
  const unsigned shift = bid_y * size; \
  if (i < size) { \
    py[i + shift] = op(pk[mbk * bid_y], px[i + mbx * shift]); \
  } \
}

#define OPENCLDEV_KERNEL_FW_AB(name, op) \
kernel void name##_fw_kernel( \
    const global float *pa, const global float *pb, const unsigned size, \
    const unsigned mba, const unsigned mbb, global float *py) { \
  const unsigned i = get_global_id(0); \
  const unsigned bid_y = get_group_id(1); \
  const unsigned shift = bid_y * size; \
  if (i < size) { \
    py[i + shift] = op(pa[i + mba * shift], pb[i + mbb * shift]); \
  } \
}
