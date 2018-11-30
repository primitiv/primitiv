#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(sin, sin(px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(sin, cos(px[i]) * pgy[i], GROUP_SIZE)
