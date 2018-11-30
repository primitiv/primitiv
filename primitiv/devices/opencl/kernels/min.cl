#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void min_fw_kernel(
    const global float *px, const unsigned skip,
    const unsigned n, global float *py) {
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  float thread_min = INFINITY;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    thread_min = min(px[i * skip], thread_min);
  }
  temp[tid] = thread_min;
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (unsigned k = GROUP_SIZE / 2; k > 0; k >>= 1) {
    if (tid < k) temp[tid] = min(temp[tid + k], temp[tid]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) py[bid] = temp[0];
}

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void min_bw_kernel(
    const global float *px, const global float *py,
    const global float *pgy, const unsigned skip,
    const unsigned n, global float *pgx) {
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  const float min_val = py[bid];
  local unsigned argmin_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  pgx += bid % skip + (bid / skip) * skip * n;
  unsigned thread_argmin = n;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    if (px[i * skip] == min_val) {
      thread_argmin = min(i, thread_argmin);
    }
  }
  argmin_val[tid] = thread_argmin;
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (unsigned k = GROUP_SIZE / 2; k > 0; k >>= 1) {
    if (tid < k) argmin_val[tid] = min(argmin_val[tid + k], argmin_val[tid]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) pgx[argmin_val[0] * skip] += pgy[bid];
}
