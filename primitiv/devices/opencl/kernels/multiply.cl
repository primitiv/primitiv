inline float inline_mul(const float a, const float b) { return a * b; }

OPENCLDEV_KERNEL_FW_X_CONST(multiply_const, px[i] * k)
OPENCLDEV_KERNEL_BW_X_CONST(multiply_const, k * pgy[i])
OPENCLDEV_KERNEL_FW_X_SCALAR_R(multiply_scalar, inline_mul)
OPENCLDEV_KERNEL_FW_AB(multiply, inline_mul)

kernel void multiply_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    atomic_add_float(pga + a_ofs, gy * pb[b_ofs]);
    atomic_add_float(pgb + b_ofs, gy * pa[a_ofs]);
  }
}
