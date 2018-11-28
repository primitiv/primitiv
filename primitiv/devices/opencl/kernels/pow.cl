OPENCLDEV_KERNEL_FW_X_CONST(pow_const_r, pow(px[i], k))
OPENCLDEV_KERNEL_FW_X_CONST(pow_const_l, pow(k, px[i]))
OPENCLDEV_KERNEL_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i])
OPENCLDEV_KERNEL_BW_X_CONST(pow_const_l, log(k) * pgy[i] * py[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(pow_scalar_r, pow)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(pow_scalar_l, pow)
OPENCLDEV_KERNEL_FW_AB(pow, pow)

kernel void pow_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] * py[y_ofs];
    atomic_add_float(pga + a_ofs, k * pb[b_ofs] / pa[a_ofs]);
    atomic_add_float(pgb + b_ofs, k * log(pa[a_ofs]));
  }
}
