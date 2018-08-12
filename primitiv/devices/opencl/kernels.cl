inline float inline_add(const float a, const float b) { return a + b; }
inline float inline_sub(const float a, const float b) { return a - b; }
inline float inline_mul(const float a, const float b) { return a * b; }
inline float inline_div(const float a, const float b) { return a / b; }

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

kernel void set_identity_kernel(
    const unsigned size, const unsigned skip, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = !(i % skip);
}

kernel void pick_fw_kernel(
    const global float *px, const global unsigned *pi,
    const unsigned wx, const unsigned wy, const unsigned sx,
    const unsigned si, const unsigned sy, global float *py) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  const unsigned oy = bid_y * sy;
  if (t < sy) py[oy + t] = px[ox + (t / wy) * wx + (t % wy)];
}

kernel void slice_fw_kernel(
    const global float *px, const unsigned shift, const unsigned span,
    const unsigned skip, const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[(i / span) * skip + (i % span) + shift];
}

kernel void concat_fw_kernel(
    const global float *px, const unsigned span, const unsigned skip,
    const unsigned x_size, const unsigned y_size,
    global float *py, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < y_size) py[(i / span) * skip + (i % span) + shift] = px[i % x_size];
}

inline void atomic_add_float(global float *source, const float operand) {
  union {
    unsigned u;
    float f;
  } oldval, newval;
  unsigned readback;
  oldval.f = *source;
  newval.f = oldval.f + operand;
  while ((readback = atomic_cmpxchg(
      (global unsigned *) source, oldval.u, newval.u)) != oldval.u) {
    oldval.u = readback;
    newval.f = oldval.f + operand;
  }
}

kernel void pick_bw_kernel(
    const global float *pgy, const global unsigned *pi,
    const unsigned wx, const unsigned wy,
    const unsigned sx, const unsigned si, const unsigned sy,
    global float *pgx) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  const unsigned oy = bid_y * sy;
  if (t < sy) {
    atomic_add_float(pgx + ox + (t / wy) * wx + (t % wy), pgy[oy + t]);
  }
}

kernel void slice_bw_kernel(
    const global float *pgy, const unsigned wx, const unsigned wy,
    const unsigned nx, const unsigned ny,
    global float *pgx, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < wy * max(nx, ny)) {
    atomic_add_float(
        pgx + shift + ((i / wy) * wx + (i % wy)) % (wx * nx),
        pgy[i % (wy * ny)]);
  }
}

#define OPENCLDEV_KERNEL_FW_X(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const unsigned size, global float *py) { \
    const unsigned i = get_global_id(0); \
  if (i < size) py[i] = (op); \
}

#define OPENCLDEV_KERNEL_BW_X(name, op) \
kernel void name##_bw_kernel( \
    const global float *px, const global float *py, const global float *pgy, \
    const unsigned size, global float *pgx) { \
   const unsigned i = get_global_id(0); \
   if (i < size) pgx[i] += (op); \
}

#define OPENCLDEV_KERNEL_FW_X_CONST(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const float k, \
    const unsigned size, global float *py) { \
  const unsigned i = get_global_id(0); \
  if (i < size) py[i] = (op); \
 }

#define OPENCLDEV_KERNEL_BW_X_CONST(name, op) \
kernel void name##_bw_kernel( \
    const global float *px, const global float *py, const global float *pgy, \
    const float k, const unsigned size, global float *pgx) { \
  const unsigned i = get_global_id(0); \
  if (i < size) pgx[i] += (op); \
}

#define OPENCLDEV_KERNEL_FW_X_SCALAR_R(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const global float *pk, const unsigned size, \
    const unsigned mbx, const unsigned mbk, global float *py) { \
  const unsigned i = get_global_id(0); \
  const unsigned bid_y = get_group_id(1); \
  const unsigned shift = bid_y * size; \
  if (i < size) { \
    py[i + shift] = op(px[i + mbx * shift], pk[mbk * bid_y]); \
  } \
}

#define OPENCLDEV_KERNEL_FW_X_SCALAR_L(name, op) \
kernel void name##_fw_kernel( \
    const global float *px, const global float *pk, const unsigned size, \
    const unsigned mbx, const unsigned mbk, global float *py) { \
  const unsigned i = get_global_id(0); \
  const unsigned bid_y = get_group_id(1); \
  const unsigned shift = bid_y * size; \
  if (i < size) { \
    py[i + shift] = op(pk[mbk * bid_y], px[i + mbx * shift]); \
  } \
}

#define OPENCLDEV_KERNEL_FW_AB(name, op) \
kernel void name##_fw_kernel( \
    const global float *pa, const global float *pb, const unsigned size, \
    const unsigned mba, const unsigned mbb, global float *py) { \
  const unsigned i = get_global_id(0); \
  const unsigned bid_y = get_group_id(1); \
  const unsigned shift = bid_y * size; \
  if (i < size) { \
    py[i + shift] = op(pa[i + mba * shift], pb[i + mbb * shift]); \
  } \
}

OPENCLDEV_KERNEL_FW_X(negate, -px[i])
OPENCLDEV_KERNEL_FW_X(abs, fabs(px[i]))
OPENCLDEV_KERNEL_FW_X(sqrt, sqrt(px[i]))
OPENCLDEV_KERNEL_FW_X(exp, exp(px[i]))
OPENCLDEV_KERNEL_FW_X(log, log(px[i]))
OPENCLDEV_KERNEL_FW_X(tanh, tanh(px[i]))
OPENCLDEV_KERNEL_FW_X(sigmoid, .5f + .5f  * tanh(.5f * px[i]))
OPENCLDEV_KERNEL_FW_X(
    softplus, max(px[i], .0f) + log(1.f + exp(-fabs(px[i]))))
OPENCLDEV_KERNEL_FW_X(sin, sin(px[i]))
OPENCLDEV_KERNEL_FW_X(cos, cos(px[i]))
OPENCLDEV_KERNEL_FW_X(tan, tan(px[i]))

OPENCLDEV_KERNEL_BW_X(abs, sign(px[i]) * pgy[i])
OPENCLDEV_KERNEL_BW_X(sqrt, .5f * pgy[i] / py[i])
OPENCLDEV_KERNEL_BW_X(exp, py[i] * pgy[i])
OPENCLDEV_KERNEL_BW_X(log, pgy[i] / px[i])
OPENCLDEV_KERNEL_BW_X(tanh, (1.f - py[i] * py[i]) * pgy[i])
OPENCLDEV_KERNEL_BW_X(sigmoid, py[i] * (1.f - py[i]) * pgy[i])
OPENCLDEV_KERNEL_BW_X(softplus, (.5f + .5f * tanh(.5f * px[i])) * pgy[i])
OPENCLDEV_KERNEL_BW_X(sin, cos(px[i]) * pgy[i])
OPENCLDEV_KERNEL_BW_X(cos, -sin(px[i]) * pgy[i])
OPENCLDEV_KERNEL_BW_X(tan, (1.f + py[i] * py[i]) * pgy[i])

OPENCLDEV_KERNEL_FW_X_CONST(add_const, px[i] + k)
OPENCLDEV_KERNEL_FW_X_CONST(subtract_const_r, px[i] - k)
OPENCLDEV_KERNEL_FW_X_CONST(subtract_const_l, k - px[i])
OPENCLDEV_KERNEL_FW_X_CONST(multiply_const, px[i] * k)
OPENCLDEV_KERNEL_FW_X_CONST(divide_const_r, px[i] / k)
OPENCLDEV_KERNEL_FW_X_CONST(divide_const_l, k / px[i])
OPENCLDEV_KERNEL_FW_X_CONST(pow_const_r, pow(px[i], k))
OPENCLDEV_KERNEL_FW_X_CONST(pow_const_l, pow(k, px[i]))
OPENCLDEV_KERNEL_FW_X_CONST(prelu, max(px[i], .0f) + k * min(px[i], .0f))
OPENCLDEV_KERNEL_FW_X_CONST(
    elu, max(px[i], .0f) + k * (exp(min(px[i], .0f)) - 1.0f))

kernel void pown_fw_kernel(
    const global float *px, const int k,
    const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = pown(px[i], k);
 }

OPENCLDEV_KERNEL_BW_X_CONST(add_const, pgy[i])
OPENCLDEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i])
OPENCLDEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i])
OPENCLDEV_KERNEL_BW_X_CONST(multiply_const, k * pgy[i])
OPENCLDEV_KERNEL_BW_X_CONST(divide_const_r, pgy[i] / k)
OPENCLDEV_KERNEL_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i])
OPENCLDEV_KERNEL_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i])
OPENCLDEV_KERNEL_BW_X_CONST(pow_const_l, log(k) * pgy[i] * py[i])
OPENCLDEV_KERNEL_BW_X_CONST(
    prelu, pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f)))
OPENCLDEV_KERNEL_BW_X_CONST(
    elu, pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f)))

kernel void pown_bw_kernel(
    const global float *px, const global float *py, const global float *pgy,
    const int k, const unsigned size, global float *pgx) {
  const unsigned i = get_global_id(0);
  if (i < size) pgx[i] += k * pgy[i] * py[i] / px[i];
}

OPENCLDEV_KERNEL_FW_X_SCALAR_R(add_scalar, inline_add)
OPENCLDEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, inline_sub)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, inline_sub)
OPENCLDEV_KERNEL_FW_X_SCALAR_R(multiply_scalar, inline_mul)
OPENCLDEV_KERNEL_FW_X_SCALAR_R(divide_scalar_r, inline_div)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(divide_scalar_l, inline_div)
OPENCLDEV_KERNEL_FW_X_SCALAR_R(pow_scalar_r, pow)
OPENCLDEV_KERNEL_FW_X_SCALAR_L(pow_scalar_l, pow)

OPENCLDEV_KERNEL_FW_AB(add, inline_add)
OPENCLDEV_KERNEL_FW_AB(subtract, inline_sub)
OPENCLDEV_KERNEL_FW_AB(multiply, inline_mul)
OPENCLDEV_KERNEL_FW_AB(divide, inline_div)
OPENCLDEV_KERNEL_FW_AB(pow, pow)

#undef OPENCLDEV_KERNEL_FW_X
#undef OPENCLDEV_KERNEL_BW_X
#undef OPENCLDEV_KERNEL_FW_X_CONST
#undef OPENCLDEV_KERNEL_BW_X_CONST
#undef CUDADEV_KERNEL_FW_X_SCALAR_R
#undef CUDADEV_KERNEL_FW_X_SCALAR_L
#undef CUDADEV_KERNEL_FW_AB

kernel void add_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    atomic_add_float(pga + i + mba * shift, gy);
    atomic_add_float(pgb + i + mbb * shift, gy);
  }
}

kernel void subtract_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    atomic_add_float(pga + i + mba * shift, gy);
    atomic_add_float(pgb + i + mbb * shift, -gy);
  }
}

kernel void multiply_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    atomic_add_float(pga + a_ofs, gy * pb[b_ofs]);
    atomic_add_float(pgb + b_ofs, gy * pa[a_ofs]);
  }
}

kernel void divide_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    atomic_add_float(pga + i + mba * shift, k);
    atomic_add_float(pgb + b_ofs, -k * py[y_ofs]);
  }
}

kernel void pow_bw_kernel(
    const global float *pa, const global float *pb,
    const global float *py, const global float *pgy,
    const unsigned size, const unsigned mba, const unsigned mbb,
    global float *pga, global float *pgb) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) {
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] * py[y_ofs];
    atomic_add_float(pga + a_ofs, k * pb[b_ofs] / pa[a_ofs]);
    atomic_add_float(pgb + b_ofs, k * log(pa[a_ofs]));
  }
}

kernel void transpose_fw_kernel(
    const global float *px, unsigned rows, unsigned cols, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}

kernel void transpose_bw_kernel(
    const global float *py, const unsigned rows, const unsigned cols,
    global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}

kernel void flip_fw_kernel(
    const global float *px, unsigned skip, unsigned n, unsigned r, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned offset = j * n - j % skip * (n - 1);
  if (i < n && j < r) {
    py[offset + i * skip] = px[offset + (n - i - 1) * skip];
  }
}

kernel void flip_bw_kernel(
    const global float *py, unsigned skip, unsigned n, unsigned r, global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned offset = j * n - j % skip * (n - 1);
  if (i < n && j < r) {
    px[offset + i * skip] += py[offset + (n - i - 1) * skip];
  }
}

kernel void permute_dims_fw_kernel(
    const global float *px, const unsigned ndims, constant unsigned *x_strides,
    constant unsigned *y_strides, const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_z = get_group_id(1);
  const unsigned ofs = bid_z * size;
  if (i < size) {
    unsigned tmp = i;
    unsigned j = 0;
    // TODO(vbkaisetsu):
    // Implove implementation
    for (unsigned d = 0; d < ndims; ++d) {
      const unsigned p = tmp / x_strides[d];
      tmp -= p * x_strides[d];
      j += p * y_strides[d];
    }
    py[ofs + j] = px[ofs + i];
  }
}

kernel void permute_dims_bw_kernel(
    const global float *py, const unsigned ndims, constant unsigned *x_strides,
    constant unsigned *y_strides, const unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned bid_z = get_group_id(1);
  const unsigned ofs = bid_z * size;
  if (i < size) {
    unsigned tmp = i;
    unsigned j = 0;
    // TODO(vbkaisetsu):
    // Implove implementation
    for (unsigned d = 0; d < ndims; ++d) {
      const unsigned p = tmp / x_strides[d];
      tmp -= p * x_strides[d];
      j += p * y_strides[d];
    }
    px[ofs + i] += py[ofs + j];
  }
}

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

kernel void broadcast_fw_kernel(
    const global float *px, const unsigned skip1, const unsigned skip2,
    const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}

kernel void batch_pick_fw_kernel(
    const global float *px, const global unsigned *pi,
    const unsigned si, const unsigned sy, global float *py) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = pi[bid_y * si] * sy;
  const unsigned oy = bid_y * sy;
  if (t < sy) py[oy + t] = px[ox + t];
}

kernel void batch_slice_fw_kernel(
    const global float *px, const unsigned shift, const unsigned size,
    global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i + shift];
}

kernel void batch_concat_fw_kernel(
    const global float *px, const unsigned y_size,
    global float *py, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < y_size) py[i + shift] = px[i];
}

kernel void batch_sum_fw_kernel(
    const global float *px, const unsigned size,
    const unsigned batch, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) {
    float temp = .0f;
    px += i;
    for (unsigned j = 0; j < batch; ++j, px += size) {
      temp += *px;
    }
    py[i] = temp;
  }
}

kernel void batch_pick_bw_kernel(
    const global float *pgy, const global unsigned *pi,
    const unsigned si, const unsigned sy, global float *pgx) {
  const unsigned t = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned ox = pi[bid_y * si] * sy;
  const unsigned oy = bid_y * sy;
  if (t < sy) atomic_add_float(pgx + ox + t, pgy[oy + t]);
}

kernel void batch_slice_bw_kernel(
    const global float *pgy, const unsigned size,
    global float *pgx, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < size) pgx[i + shift] += pgy[i];
}

kernel void inplace_multiply_const_kernel(
    const float k, const unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  if (i < size) px[i] *= k;
}

kernel void inplace_add_kernel(
    const global float *px, const unsigned size,
    const unsigned mbx, const unsigned mby, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, px[i + mbx * shift]);
}

kernel void inplace_subtract_kernel(
    const global float *px, const unsigned size,
    const unsigned mbx, const unsigned mby, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, -px[i + mbx * shift]);
}
