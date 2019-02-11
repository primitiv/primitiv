#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] += temp[tid + k]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define SUM_FW_KERNEL(GROUP_SIZE) \
kernel void sum_fw_kernel_##GROUP_SIZE( \
    const global float *px, const unsigned skip, const unsigned n, \
    global float *py) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  local float temp[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  temp[tid] = 0; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) temp[tid] += px[i * skip]; \
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

SUM_FW_KERNEL(1024)
SUM_FW_KERNEL(512)
SUM_FW_KERNEL(256)
SUM_FW_KERNEL(128)
SUM_FW_KERNEL(64)
SUM_FW_KERNEL(32)
SUM_FW_KERNEL(16)
SUM_FW_KERNEL(8)
SUM_FW_KERNEL(4)
SUM_FW_KERNEL(2)
SUM_FW_KERNEL(1)

#undef REDUCE
