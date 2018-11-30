#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void batch_concat_fw_kernel(
    const global float *px, const unsigned y_size,
    global float *py, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < y_size) py[i + shift] = px[i];
}
