#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X_CONST(prelu, max(px[i], .0f) + k * min(px[i], .0f), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X_CONST(
    prelu, pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f)), GROUP_SIZE)
