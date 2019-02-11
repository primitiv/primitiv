kernel void batch_sum_fw_kernel(
    const global float *px, const unsigned size,
    const unsigned batch, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) {
    float temp = .0f;
    px += i;
    for (unsigned j = 0; j < batch; ++j, px += size) {
      temp += *px;
    }
    py[i] = temp;
  }
}
