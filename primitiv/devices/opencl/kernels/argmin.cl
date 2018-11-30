#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void argmin_kernel(
    const global float *px, const unsigned skip,
    const unsigned n, global unsigned *py) {
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float min_val[GROUP_SIZE];
  local unsigned argmin_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  min_val[tid] = INFINITY;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    const float val = px[i * skip];
    if (val < min_val[tid]) {
      min_val[tid] = val;
      argmin_val[tid] = i;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (int k = GROUP_SIZE / 2; k > 0; k >>= 1) {
    if (tid < k) {
      if (min_val[tid + k] < min_val[tid]
          || (min_val[tid + k] == min_val[tid]
              && argmin_val[tid + k] < argmin_val[tid])) {
        min_val[tid] = min_val[tid + k];
        argmin_val[tid] = argmin_val[tid + k];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) py[bid] = argmin_val[0];
}
