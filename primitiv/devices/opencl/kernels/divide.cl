inline float inline_div(const float a, const float b) { return a / b; }

OPENCLDEV_KERNEL_FW_X_CONST(divide_const_r, px[i] / k)
OPENCLDEV_KERNEL_FW_X_CONST(divide_const_l, k / px[i])
OPENCLDEV_KERNEL_BW_X_CONST(divide_const_r, pgy[i] / k)
OPENCLDEV_KERNEL_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(divide_scalar_r, inline_div)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(divide_scalar_l, inline_div)
OPENCLDEV_KERNEL_FW_AB(divide, inline_div)

kernel void divide_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    atomic_add_float(pga + i + mba * shift, k);
    atomic_add_float(pgb + b_ofs, -k * py[y_ofs]);
  }
}
