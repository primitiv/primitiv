kernel void batch_concat_fw_kernel(
    const global float *px, const unsigned y_size,
    global float *py, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < y_size) py[i + shift] = px[i];
}
