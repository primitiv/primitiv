#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] > max_val[tid] \
          || (max_val[tid + k] == max_val[tid] \
              && argmax_val[tid + k] < argmax_val[tid])) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define ARGMAX_KERNEL(GROUP_SIZE) \
kernel void argmax_kernel_##GROUP_SIZE( \
    const global float *px, const unsigned skip, \
    const unsigned n, global unsigned *py) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  local float max_val[GROUP_SIZE]; \
  local unsigned argmax_val[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  max_val[tid] = -INFINITY; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    const float val = px[i * skip]; \
    if (val > max_val[tid]) { \
      max_val[tid] = val; \
      argmax_val[tid] = i; \
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
  if (tid == 0) py[bid] = argmax_val[0]; \
}

ARGMAX_KERNEL(1024)
ARGMAX_KERNEL(512)
ARGMAX_KERNEL(256)
ARGMAX_KERNEL(128)
ARGMAX_KERNEL(64)
ARGMAX_KERNEL(32)
ARGMAX_KERNEL(16)
ARGMAX_KERNEL(8)
ARGMAX_KERNEL(4)
ARGMAX_KERNEL(2)
ARGMAX_KERNEL(1)

#undef REDUCE
