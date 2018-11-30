#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(log, log(px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(log, pgy[i] / px[i], GROUP_SIZE)
