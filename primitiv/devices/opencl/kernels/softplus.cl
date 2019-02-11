OPENCLDEV_KERNEL_FW_X(
    softplus, max(px[i], .0f) + log(1.f + exp(-fabs(px[i]))))
OPENCLDEV_KERNEL_BW_X(softplus, (.5f + .5f * tanh(.5f * px[i])) * pgy[i])
