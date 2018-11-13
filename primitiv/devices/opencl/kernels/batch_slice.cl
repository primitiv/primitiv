kernel void batch_slice_fw_kernel(
    const global float *px, const unsigned shift, const unsigned size,
    global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i + shift];
}

kernel void batch_slice_bw_kernel(
    const global float *pgy, const unsigned size,
    global float *pgx, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < size) pgx[i + shift] += pgy[i];
}
