inline float logsumexp2_fw_kernel(float a, float b) {
  return a > b
    ? a + log(1.f + exp(b - a))
    : b + log(1.f + exp(a - b));
}

#define REDUCE(k, GROUP_SIZE) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = logsumexp2_fw_kernel(temp[tid], temp[tid + k]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
  }

#define LOGSUMEXP_FW_KERNEL(GROUP_SIZE) \
kernel void logsumexp_fw_kernel_##GROUP_SIZE( \
    const global float *px, const unsigned skip, const unsigned n, \
    global float *py) { \
  const unsigned bid = get_group_id(0); \
  const unsigned tid = get_local_id(0); \
  local float temp[GROUP_SIZE]; \
  px += bid % skip + (bid / skip) * skip * n; \
  temp[tid] = -1e38; \
  for (unsigned i = tid; i < n; i += GROUP_SIZE) { \
    temp[tid] = logsumexp2_fw_kernel(temp[tid], px[i * skip]); \
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
  if (tid == 0) py[bid] = temp[0]; \
}

LOGSUMEXP_FW_KERNEL(1024)
LOGSUMEXP_FW_KERNEL(512)
LOGSUMEXP_FW_KERNEL(256)
LOGSUMEXP_FW_KERNEL(128)
LOGSUMEXP_FW_KERNEL(64)
LOGSUMEXP_FW_KERNEL(32)
LOGSUMEXP_FW_KERNEL(16)
LOGSUMEXP_FW_KERNEL(8)
LOGSUMEXP_FW_KERNEL(4)
LOGSUMEXP_FW_KERNEL(2)
LOGSUMEXP_FW_KERNEL(1)

#undef REDUCE
