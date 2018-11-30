#if !defined(GROUP_SIZE_X) || !defined(GROUP_SIZE_Y)
  #define GROUP_SIZE_X 8
  #define GROUP_SIZE_Y 8
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
void transpose_fw_kernel(
    const global float *px, unsigned rows, unsigned cols, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}

kernel __attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
void transpose_bw_kernel(
    const global float *py, const unsigned rows, const unsigned cols,
    global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}
