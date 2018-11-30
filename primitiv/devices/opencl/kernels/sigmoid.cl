#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(sigmoid, .5f + .5f  * tanh(.5f * px[i]), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(sigmoid, py[i] * (1.f - py[i]) * pgy[i], GROUP_SIZE)
