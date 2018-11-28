inline float inline_sub(const float a, const float b) { return a - b; }

OPENCLDEV_KERNEL_FW_X_CONST(subtract_const_r, px[i] - k)
OPENCLDEV_KERNEL_FW_X_CONST(subtract_const_l, k - px[i])
OPENCLDEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i])
OPENCLDEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, inline_sub)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, inline_sub)
OPENCLDEV_KERNEL_FW_AB(subtract, inline_sub)

kernel void subtract_bw_kernel(
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
    atomic_add_float(pgb + i + mbb * shift, -gy);
  }
}
