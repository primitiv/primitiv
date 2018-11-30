#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void sum_fw_kernel(
    const global float *px, const unsigned skip, const unsigned n,
    global float *py) {
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = 0;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) temp[tid] += px[i * skip];
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (unsigned k = GROUP_SIZE / 2; k > 0; k >>= 1) {
    if (tid < k) temp[tid] += temp[tid + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) py[bid] = temp[0];
}
