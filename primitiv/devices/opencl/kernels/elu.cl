OPENCLDEV_KERNEL_FW_X_CONST(
    elu, max(px[i], .0f) + k * (exp(min(px[i], .0f)) - 1.0f))
OPENCLDEV_KERNEL_BW_X_CONST(
    elu, pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f)))
