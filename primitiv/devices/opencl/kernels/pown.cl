kernel void pown_fw_kernel(
    const global float *px, const int k,
    const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = pown(px[i], k);
}

kernel void pown_bw_kernel(
    const global float *px, const global float *py, const global float *pgy,
    const int k, const unsigned size, global float *pgx) {
  const unsigned i = get_global_id(0);
  if (i < size) pgx[i] += k * pgy[i] * py[i] / px[i];
}
