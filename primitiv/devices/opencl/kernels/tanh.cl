OPENCLDEV_KERNEL_FW_X(tanh, tanh(px[i]))
OPENCLDEV_KERNEL_BW_X(tanh, (1.f - py[i] * py[i]) * pgy[i])
