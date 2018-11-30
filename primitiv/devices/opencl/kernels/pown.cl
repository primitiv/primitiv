#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void pown_fw_kernel(
    const global float *px, const int k,
    const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = pown(px[i], k);
}

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void pown_bw_kernel(
    const global float *px, const global float *py, const global float *pgy,
    const int k, const unsigned size, global float *pgx) {
  const unsigned i = get_global_id(0);
  if (i < size) pgx[i] += k * pgy[i] * py[i] / px[i];
}
