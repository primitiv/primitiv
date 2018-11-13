kernel void broadcast_fw_kernel(
    const global float *px, const unsigned skip1, const unsigned skip2,
    const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}
