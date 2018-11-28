#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (min_val[tid + k] < min_val[tid] \
          || (min_val[tid + k] == min_val[tid] \
              && argmin_val[tid + k] < argmin_val[tid])) { \
        min_val[tid] = min_val[tid + k]; \
        argmin_val[tid] = argmin_val[tid + k]; \
      } \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define ARGMIN_KERNEL(GROUP_SIZE) \
kernel void argmin_kernel_##GROUP_SIZE( \
    const global float *px, const unsigned skip, \
    const unsigned n, global unsigned *py) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  local float min_val[GROUP_SIZE]; \
  local unsigned argmin_val[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  min_val[tid] = INFINITY; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    const float val = px[i * skip]; \
    if (val < min_val[tid]) { \
      min_val[tid] = val; \
      argmin_val[tid] = i; \
    } \
  } \
  barrier(CLK_LOCAL_MEM_FENCE); \
  REDUCE(512, GROUP_SIZE) \
  REDUCE(256, GROUP_SIZE) \
  REDUCE(128, GROUP_SIZE) \
  REDUCE(64, GROUP_SIZE) \
  REDUCE(32, GROUP_SIZE) \
  REDUCE(16, GROUP_SIZE) \
  REDUCE(8, GROUP_SIZE) \
  REDUCE(4, GROUP_SIZE) \
  REDUCE(2, GROUP_SIZE) \
  REDUCE(1, GROUP_SIZE) \
  if (tid == 0) py[bid] = argmin_val[0]; \
}

ARGMIN_KERNEL(1024)
ARGMIN_KERNEL(512)
ARGMIN_KERNEL(256)
ARGMIN_KERNEL(128)
ARGMIN_KERNEL(64)
ARGMIN_KERNEL(32)
ARGMIN_KERNEL(16)
ARGMIN_KERNEL(8)
ARGMIN_KERNEL(4)
ARGMIN_KERNEL(2)
ARGMIN_KERNEL(1)

#undef REDUCE
