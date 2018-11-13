inline float inline_add(const float a, const float b) { return a + b; }

OPENCLDEV_KERNEL_FW_X_CONST(add_const, px[i] + k)
OPENCLDEV_KERNEL_BW_X_CONST(add_const, pgy[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(add_scalar, inline_add)
OPENCLDEV_KERNEL_FW_AB(add, inline_add)

kernel void add_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    atomic_add_float(pga + i + mba * shift, gy);
    atomic_add_float(pgb + i + mbb * shift, gy);
  }
}
