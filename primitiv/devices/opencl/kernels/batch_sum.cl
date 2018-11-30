#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void batch_sum_fw_kernel(
    const global float *px, const unsigned size,
    const unsigned batch, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) {
    float temp = .0f;
    px += i;
    for (unsigned j = 0; j < batch; ++j, px += size) {
      temp += *px;
    }
    py[i] = temp;
  }
}
