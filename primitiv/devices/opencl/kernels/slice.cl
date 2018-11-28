kernel void slice_fw_kernel(
    const global float *px, const unsigned shift, const unsigned span,
    const unsigned skip, const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[(i / span) * skip + (i % span) + shift];
}

kernel void slice_bw_kernel(
    const global float *pgy, const unsigned wx, const unsigned wy,
    const unsigned nx, const unsigned ny,
    global float *pgx, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < wy * max(nx, ny)) {
    atomic_add_float(
        pgx + shift + ((i / wy) * wx + (i % wy)) % (wx * nx),
        pgy[i % (wy * ny)]);
  }
}
