#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = min(temp[tid + k], temp[tid]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define MIN_FW_KERNEL(GROUP_SIZE) \
kernel void min_fw_kernel_##GROUP_SIZE( \
    const global float *px, const unsigned skip, \
    const unsigned n, global float *py) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  local float temp[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  float thread_max = INFINITY; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    thread_max = min(px[i * skip], thread_max); \
  } \
  temp[tid] = thread_max; \
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
  if (tid == 0) py[bid] = temp[0]; \
}

MIN_FW_KERNEL(1024)
MIN_FW_KERNEL(512)
MIN_FW_KERNEL(256)
MIN_FW_KERNEL(128)
MIN_FW_KERNEL(64)
MIN_FW_KERNEL(32)
MIN_FW_KERNEL(16)
MIN_FW_KERNEL(8)
MIN_FW_KERNEL(4)
MIN_FW_KERNEL(2)
MIN_FW_KERNEL(1)

#undef REDUCE

#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) argmin_val[tid] = min(argmin_val[tid + k], argmin_val[tid]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define MIN_BW_KERNEL(GROUP_SIZE) \
kernel void min_bw_kernel_##GROUP_SIZE( \
    const global float *px, const global float *py, \
    const global float *pgy, const unsigned skip, \
    const unsigned n, global float *pgx) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  const float min_val = py[bid]; \
  local unsigned argmin_val[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  pgx += bid % skip + (bid / skip) * skip * n; \
  unsigned thread_argmin = n; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    if (px[i * skip] == min_val) { \
      thread_argmin = min(i, thread_argmin); \
    } \
  } \
  argmin_val[tid] = thread_argmin; \
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
  if (tid == 0) pgx[argmin_val[0] * skip] += pgy[bid]; \
}

MIN_BW_KERNEL(1024)
MIN_BW_KERNEL(512)
MIN_BW_KERNEL(256)
MIN_BW_KERNEL(128)
MIN_BW_KERNEL(64)
MIN_BW_KERNEL(32)
MIN_BW_KERNEL(16)
MIN_BW_KERNEL(8)
MIN_BW_KERNEL(4)
MIN_BW_KERNEL(2)
MIN_BW_KERNEL(1)

#undef REDUCE
