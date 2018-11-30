#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(exp, exp(px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(exp, py[i] * pgy[i], GROUP_SIZE)
