#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(sqrt, sqrt(px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(sqrt, .5f * pgy[i] / py[i], GROUP_SIZE)
