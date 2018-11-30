#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void batch_pick_fw_kernel(
    const global float *px, const global unsigned *pi,
    const unsigned si, const unsigned sy, global float *py) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = pi[bid_y * si] * sy;
  const unsigned oy = bid_y * sy;
  if (t < sy) py[oy + t] = px[ox + t];
}

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void batch_pick_bw_kernel(
    const global float *pgy, const global unsigned *pi,
    const unsigned si, const unsigned sy, global float *pgx) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = pi[bid_y * si] * sy;
  const unsigned oy = bid_y * sy;
  if (t < sy) atomic_add_float(pgx + ox + t, pgy[oy + t]);
}
