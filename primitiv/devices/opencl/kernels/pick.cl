kernel void pick_fw_kernel(
    const global float *px, const global unsigned *pi,
    const unsigned wx, const unsigned wy, const unsigned sx,
    const unsigned si, const unsigned sy, global float *py) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  const unsigned oy = bid_y * sy;
  if (t < sy) py[oy + t] = px[ox + (t / wy) * wx + (t % wy)];
}

kernel void pick_bw_kernel(
    const global float *pgy, const global unsigned *pi,
    const unsigned wx, const unsigned wy,
    const unsigned sx, const unsigned si, const unsigned sy,
    global float *pgx) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  const unsigned oy = bid_y * sy;
  if (t < sy) {
    atomic_add_float(pgx + ox + (t / wy) * wx + (t % wy), pgy[oy + t]);
  }
}
