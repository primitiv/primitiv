#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = max(temp[tid + k], temp[tid]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define MAX_FW_KERNEL(GROUP_SIZE) \
kernel void max_fw_kernel_##GROUP_SIZE( \
    const global float *px, const unsigned skip, \
    const unsigned n, global float *py) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  local float temp[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  float thread_max = -INFINITY; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    thread_max = max(px[i * skip], thread_max); \
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

MAX_FW_KERNEL(1024)
MAX_FW_KERNEL(512)
MAX_FW_KERNEL(256)
MAX_FW_KERNEL(128)
MAX_FW_KERNEL(64)
MAX_FW_KERNEL(32)
MAX_FW_KERNEL(16)
MAX_FW_KERNEL(8)
MAX_FW_KERNEL(4)
MAX_FW_KERNEL(2)
MAX_FW_KERNEL(1)

#undef REDUCE

#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) argmax_val[tid] = min(argmax_val[tid + k], argmax_val[tid]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define MAX_BW_KERNEL(GROUP_SIZE) \
kernel void max_bw_kernel_##GROUP_SIZE( \
    const global float *px, const global float *py, \
    const global float *pgy, const unsigned skip, \
    const unsigned n, global float *pgx) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  const float max_val = py[bid]; \
  local unsigned argmax_val[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  pgx += bid % skip + (bid / skip) * skip * n; \
  unsigned thread_argmax = n; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    if (px[i * skip] == max_val) { \
      thread_argmax = min(i, thread_argmax); \
    } \
  } \
  argmax_val[tid] = thread_argmax; \
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
  if (tid == 0) pgx[argmax_val[0] * skip] += pgy[bid]; \
}

MAX_BW_KERNEL(1024)
MAX_BW_KERNEL(512)
MAX_BW_KERNEL(256)
MAX_BW_KERNEL(128)
MAX_BW_KERNEL(64)
MAX_BW_KERNEL(32)
MAX_BW_KERNEL(16)
MAX_BW_KERNEL(8)
MAX_BW_KERNEL(4)
MAX_BW_KERNEL(2)
MAX_BW_KERNEL(1)

#undef REDUCE
