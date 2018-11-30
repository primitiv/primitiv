#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(cos, cos(px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(cos, -sin(px[i]) * pgy[i], GROUP_SIZE)
