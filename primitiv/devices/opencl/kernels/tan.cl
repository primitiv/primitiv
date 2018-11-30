#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(tan, tan(px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(tan, (1.f + py[i] * py[i]) * pgy[i], GROUP_SIZE)
