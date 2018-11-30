#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

inline float inline_sub(const float a, const float b) { return a - b; }

OPENCLDEV_KERNEL_FW_X_CONST(subtract_const_r, px[i] - k, GROUP_SIZE)
OPENCLDEV_KERNEL_FW_X_CONST(subtract_const_l, k - px[i], GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i], GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i], GROUP_SIZE)
OPENCLDEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, inline_sub, GROUP_SIZE)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, inline_sub, GROUP_SIZE)
OPENCLDEV_KERNEL_FW_AB(subtract, inline_sub, GROUP_SIZE)

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void subtract_bw_kernel(
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
