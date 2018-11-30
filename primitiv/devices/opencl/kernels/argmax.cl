#ifndef GROUP_SIZE
  #define GROUP_SIZE 64
#endif

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void argmax_kernel(
    const global float *px, const unsigned skip,
    const unsigned n, global unsigned *py) {
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float max_val[GROUP_SIZE];
  local unsigned argmax_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = -INFINITY;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    const float val = px[i * skip];
    if (val > max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  #pragma unroll
  for (int k = GROUP_SIZE / 2; k > 0; k >>= 1) {
    if (tid < k) {
      if (max_val[tid + k] > max_val[tid]
          || (max_val[tid + k] == max_val[tid]
              && argmax_val[tid + k] < argmax_val[tid])) {
        max_val[tid] = max_val[tid + k];
        argmax_val[tid] = argmax_val[tid + k];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) py[bid] = argmax_val[0];
}

