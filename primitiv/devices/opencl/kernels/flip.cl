kernel void flip_fw_kernel(
    const global float *px, unsigned skip, unsigned n, unsigned r, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned offset = j * n - j % skip * (n - 1);
  if (i < n && j < r) {
    py[offset + i * skip] = px[offset + (n - i - 1) * skip];
  }
}

kernel void flip_bw_kernel(
    const global float *py, unsigned skip, unsigned n, unsigned r, global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned offset = j * n - j % skip * (n - 1);
  if (i < n && j < r) {
    px[offset + i * skip] += py[offset + (n - i - 1) * skip];
  }
}
