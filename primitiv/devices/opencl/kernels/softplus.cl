#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

OPENCLDEV_KERNEL_FW_X(
    softplus, max(px[i], .0f) + log(1.f + exp(-fabs(px[i]))), GROUP_SIZE)
OPENCLDEV_KERNEL_BW_X(softplus, (.5f + .5f * tanh(.5f * px[i])) * pgy[i], GROUP_SIZE)
