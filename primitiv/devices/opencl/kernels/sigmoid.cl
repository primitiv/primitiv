OPENCLDEV_KERNEL_FW_X(sigmoid, .5f + .5f  * tanh(.5f * px[i]))
OPENCLDEV_KERNEL_BW_X(sigmoid, py[i] * (1.f - py[i]) * pgy[i])
