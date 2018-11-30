#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void batch_slice_fw_kernel(
    const global float *px, const unsigned shift, const unsigned size,
    global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i + shift];
}

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void batch_slice_bw_kernel(
    const global float *pgy, const unsigned size,
    global float *pgx, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < size) pgx[i + shift] += pgy[i];
}
