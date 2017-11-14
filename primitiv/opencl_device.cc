#include <config.h>

#include <CL/cl.hpp>

#include <iostream>
#include <random>
#include <primitiv/error.h>
#include <primitiv/opencl_device.h>

namespace primitiv {
namespace devices {

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define CDATA(x) *(static_cast<const cl::Buffer *>((x).data()))

#define SET_ARG_HOST_SCALAR(kernel, idx, type, var) \
  cl::Buffer opencl_mem_##var = cl::Buffer(context_, \
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
      sizeof(type), &var, &error); \
  kernel.setArg(idx, opencl_mem_##var);

#define SET_ARG_HOST_VECTOR(kernel, idx, type, var) \
  cl::Buffer opencl_mem_##var = cl::Buffer(context_, \
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
      sizeof(type) * var.size(), const_cast<type*>(var.data()), &error); \
  kernel.setArg(idx, opencl_mem_##var);

std::string OpenCL::kernel_code_generator() {
  std::ostringstream ss;
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void argmax_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global unsigned *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  local float max_val[GROUP_SIZE];
  local unsigned argmax_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = -1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    float val = px[i * skip];
    if (val > max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] > max_val[tid]) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = argmax_val[0];
#undef GROUP_SIZE
}
)EOS";
  }
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void argmin_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global unsigned *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  local float max_val[GROUP_SIZE];
  local unsigned argmax_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = 1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    float val = px[i * skip];
    if (val < max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] < max_val[tid]) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = argmax_val[0];
#undef GROUP_SIZE
}
)EOS";
  }
  ss << R"EOS(
kernel void set_identity_kernel(constant unsigned *size_p, constant unsigned *skip_p, global float *py) {
  unsigned i = get_global_id(0);
  unsigned size = size_p[0];
  unsigned skip = skip_p[0];
  if (i < size) py[i] = !(i % skip);
}
)EOS";
  ss << R"EOS(
kernel void pick_fw_kernel(constant float *px, constant unsigned *pi,
                           constant unsigned *wx_p, constant unsigned *wy_p, constant unsigned *sx_p,
                           constant unsigned *si_p, constant unsigned *sy_p, global float *py) {
  unsigned wx = wx_p[0];
  unsigned wy = wy_p[0];
  unsigned sx = sx_p[0];
  unsigned si = si_p[0];
  unsigned sy = sy_p[0];
  unsigned t = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  unsigned oy = bid_y * sy;
  if (t < sy) py[oy + t] = px[ox + (t / wy) * wx + (t % wy)];
}
)EOS";
  ss << R"EOS(
kernel void slice_fw_kernel(constant float *px, constant unsigned *shift_p, constant unsigned *span_p,
                            constant unsigned *skip_p, constant unsigned *size_p, global float *py) {
  unsigned span = span_p[0];
  unsigned skip = skip_p[0];
  unsigned size = size_p[0];
  unsigned i = get_global_id(0);
  if (i < size) py[i] = px[(i / span) * skip + (i % span) + shift_p[0]];
}
)EOS";
  ss << R"EOS(
kernel void concat_fw_kernel(constant float *px, constant unsigned *span_p, constant unsigned *skip_p,
                             constant unsigned *x_size_p, constant unsigned *y_size_p, global float *py, constant unsigned *shift_p) {
  unsigned span = span_p[0];
  unsigned skip = skip_p[0];
  unsigned x_size = x_size_p[0];
  unsigned y_size = y_size_p[0];
  unsigned i = get_global_id(0);
  if (i < y_size) py[(i / span) * skip + (i % span) + shift_p[0]] = px[i % x_size];
}
)EOS";
  ss << R"EOS(
inline void atomic_add_float(volatile global float *source, const float operand) {
  union {
    unsigned i;
    float f;
  } nval;
  union {
    unsigned i;
    float f;
  } pval;
  do {
    pval.f = *source;
    nval.f = pval.f + operand;
  } while (atomic_cmpxchg((volatile global unsigned *) source, pval.i, nval.i) != pval.i);
}
)EOS";
  ss << R"EOS(
kernel void pick_bw_kernel(constant float *pgy, constant unsigned *pi, constant unsigned *wx_p, constant unsigned *wy_p,
                           constant unsigned *sx_p, constant unsigned *si_p, constant unsigned *sy_p, volatile global float *pgx) {
  unsigned wx = wx_p[0];
  unsigned wy = wy_p[0];
  unsigned sx = sx_p[0];
  unsigned si = si_p[0];
  unsigned sy = sy_p[0];
  unsigned t = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  unsigned oy = bid_y * sy;
  if (t < sy) atomic_add_float(pgx + ox + (t / wy) * wx + (t % wy), pgy[oy + t]);
}
)EOS";
  ss << R"EOS(
kernel void slice_bw_kernel(constant float *pgy, constant unsigned *wx_p, constant unsigned *wy_p,
                            constant unsigned *nx_p, constant unsigned *ny_p, volatile global float *pgx, constant unsigned *shift_p) {
  unsigned wx = wx_p[0];
  unsigned wy = wy_p[0];
  unsigned nx = nx_p[0];
  unsigned ny = ny_p[0];
  unsigned shift = shift_p[0];
  unsigned i = get_global_id(0);
  if (i < wy * max(nx, ny)) atomic_add_float(pgx + shift + ((i / wy) * wx + (i % wy)) % (wx * nx), pgy[i % (wy * ny)]);
}
)EOS";

#define OPENCLDEV_KERNEL_FW_X(name, op) \
  ss << "kernel void " << name << "_fw_kernel(constant float *px, constant unsigned *size_p, global float *py) {" \
        "  unsigned size = size_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  if (i < size) py[i] = (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_BW_X(name, op) \
  ss << "kernel void " << name << "_bw_kernel(constant float *px, constant float *py, constant float *pgy," \
        "                                     constant unsigned *size_p, global float *pgx) {" \
        "  unsigned size = size_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  if (i < size) pgx[i] += (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_X_CONST(name, op) \
  ss << "kernel void " << name << "_fw_kernel(constant float *px, constant float *k_p," \
        "                                     constant unsigned *size_p, global float *py) {" \
        "  float k = k_p[0];" \
        "  unsigned size = size_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  if (i < size) py[i] = (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_BW_X_CONST(name, op) \
  ss << "kernel void " << name << "_bw_kernel(constant float *px, constant float *py, constant float *pgy," \
        "                                     constant float *k_p, constant unsigned *size_p, global float *pgx) {" \
        "  float k = k_p[0];" \
        "  unsigned size = size_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  if (i < size) pgx[i] += (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_X_SCALAR_R_INFIX(name, op) \
  ss << "kernel void " << name << "_fw_kernel(constant float *px, constant float *pk, constant unsigned *size_p," \
        "                                     constant unsigned *mbx_p, constant unsigned *mbk_p, global float *py) {" \
        "  unsigned size = size_p[0];" \
        "  unsigned mbx = mbx_p[0];" \
        "  unsigned mbk = mbk_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  unsigned bid_y = get_group_id(1);" \
        "  unsigned shift = bid_y * size;" \
        "  if (i < size) py[i + shift] = px[i + mbx * shift] " << op << " pk[mbk * bid_y];" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_X_SCALAR_L_INFIX(name, op) \
  ss << "kernel void " << name << "_fw_kernel(constant float *px, constant float *pk, constant unsigned *size_p," \
        "                                     constant unsigned *mbx_p, constant unsigned *mbk_p, global float *py) {" \
        "  unsigned size = size_p[0];" \
        "  unsigned mbx = mbx_p[0];" \
        "  unsigned mbk = mbk_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  unsigned bid_y = get_group_id(1);" \
        "  unsigned shift = bid_y * size;" \
        "  if (i < size) py[i + shift] = pk[mbk * bid_y] " << op << " px[i + mbx * shift];" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_AB_INFIX(name, op) \
  ss << "kernel void " << name << "_fw_kernel(constant float *pa, constant float *pb, constant unsigned *size_p," \
        "                                  constant unsigned *mba_p, constant unsigned *mbb_p, global float *py) {" \
        "  unsigned size = size_p[0];" \
        "  unsigned mba = mba_p[0];" \
        "  unsigned mbb = mbb_p[0];" \
        "  unsigned i = get_global_id(0);" \
        "  unsigned bid_y = get_group_id(1);" \
        "  unsigned shift = bid_y * size;" \
        "  if (i < size) py[i + shift] = pa[i + mba * shift] " << op << " pb[i + mbb * shift];" \
        "}\n";

OPENCLDEV_KERNEL_FW_X("negate", "-px[i]");
OPENCLDEV_KERNEL_FW_X("sqrt", "sqrt(px[i])");
OPENCLDEV_KERNEL_FW_X("exp", "exp(px[i])");
OPENCLDEV_KERNEL_FW_X("log", "log(px[i])");
OPENCLDEV_KERNEL_FW_X("tanh", "tanh(px[i])");
OPENCLDEV_KERNEL_FW_X("sigmoid", ".5f + .5f * tanh(.5f * px[i])");
OPENCLDEV_KERNEL_FW_X("softplus", "max(px[i], .0f) + log(1.f + exp(-fabs(px[i])))");
OPENCLDEV_KERNEL_FW_X("sin", "sin(px[i])");
OPENCLDEV_KERNEL_FW_X("cos", "cos(px[i])");
OPENCLDEV_KERNEL_FW_X("tan", "tan(px[i])");

OPENCLDEV_KERNEL_BW_X("sqrt", ".5f * pgy[i] / py[i]");
OPENCLDEV_KERNEL_BW_X("exp", "py[i] * pgy[i]");
OPENCLDEV_KERNEL_BW_X("log", "pgy[i] / px[i]");
OPENCLDEV_KERNEL_BW_X("tanh", "(1.f - py[i] * py[i]) * pgy[i]");
OPENCLDEV_KERNEL_BW_X("sigmoid", "py[i] * (1.f - py[i]) * pgy[i]");
OPENCLDEV_KERNEL_BW_X("softplus", "(.5f + .5f * tanh(.5f * px[i])) * pgy[i]");
OPENCLDEV_KERNEL_BW_X("sin", "cos(px[i]) * pgy[i]");
OPENCLDEV_KERNEL_BW_X("cos", "-sin(px[i]) * pgy[i]");
OPENCLDEV_KERNEL_BW_X("tan", "(1.f + py[i] * py[i]) * pgy[i]");

OPENCLDEV_KERNEL_FW_X_CONST("add_const", "px[i] + k");
OPENCLDEV_KERNEL_FW_X_CONST("subtract_const_r", "px[i] - k");
OPENCLDEV_KERNEL_FW_X_CONST("subtract_const_l", "k - px[i]");
OPENCLDEV_KERNEL_FW_X_CONST("multiply_const", "px[i] * k");
OPENCLDEV_KERNEL_FW_X_CONST("divide_const_r", "px[i] / k");
OPENCLDEV_KERNEL_FW_X_CONST("divide_const_l", "k / px[i]");
OPENCLDEV_KERNEL_FW_X_CONST("prelu", "max(px[i], .0f) + k * min(px[i], .0f)");
OPENCLDEV_KERNEL_FW_X_CONST("elu", "max(px[i], .0f) + k * (exp(min(px[i], .0f)) - 1.0f)");

OPENCLDEV_KERNEL_BW_X_CONST("add_const", "pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("subtract_const_r", "pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("subtract_const_l", "-pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("multiply_const", "k * pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("divide_const_r", "pgy[i] / k");
OPENCLDEV_KERNEL_BW_X_CONST("divide_const_l", "-py[i] * pgy[i] / px[i]");
OPENCLDEV_KERNEL_BW_X_CONST("prelu", "pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f))");
OPENCLDEV_KERNEL_BW_X_CONST("elu", "pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f))");

OPENCLDEV_KERNEL_FW_X_SCALAR_R_INFIX("add_scalar", "+");
OPENCLDEV_KERNEL_FW_X_SCALAR_R_INFIX("subtract_scalar_r", "-");
OPENCLDEV_KERNEL_FW_X_SCALAR_L_INFIX("subtract_scalar_l", "-");
OPENCLDEV_KERNEL_FW_X_SCALAR_R_INFIX("multiply_scalar", "*");
OPENCLDEV_KERNEL_FW_X_SCALAR_R_INFIX("divide_scalar_r", "/");
OPENCLDEV_KERNEL_FW_X_SCALAR_L_INFIX("divide_scalar_l", "/");

OPENCLDEV_KERNEL_FW_AB_INFIX("add", "+");
OPENCLDEV_KERNEL_FW_AB_INFIX("subtract", "-");
OPENCLDEV_KERNEL_FW_AB_INFIX("multiply", "*");
OPENCLDEV_KERNEL_FW_AB_INFIX("divide", "/");

#undef OPENCLDEV_KERNEL_FW_X
#undef OPENCLDEV_KERNEL_BW_X
#undef OPENCLDEV_KERNEL_FW_X_CONST
#undef OPENCLDEV_KERNEL_BW_X_CONST
#undef CUDADEV_KERNEL_FW_X_SCALAR_R
#undef CUDADEV_KERNEL_FW_X_SCALAR_L
#undef CUDADEV_KERNEL_FW_AB

  ss << R"EOS(
kernel void add_bw_kernel(constant float *pa, constant float *pb, constant float *py, constant float *pgy,
                          constant unsigned *size_p, constant unsigned *mba_p, constant unsigned *mbb_p, global float *pga, global float *pgb) {
  unsigned size = size_p[0];
  unsigned mba = mba_p[0];
  unsigned mbb = mbb_p[0];
  unsigned i = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned shift = bid_y * size;
  if (i < size) {
    float gy = pgy[i + shift];
    atomic_add_float(pga + i + mba * shift, gy);
    atomic_add_float(pgb + i + mbb * shift, gy);
  }
}
)EOS";

  ss << R"EOS(
kernel void subtract_bw_kernel(constant float *pa, constant float *pb, constant float *py, constant float *pgy,
                               constant unsigned *size_p, constant unsigned *mba_p, constant unsigned *mbb_p, global float *pga, global float *pgb) {
  unsigned size = size_p[0];
  unsigned mba = mba_p[0];
  unsigned mbb = mbb_p[0];
  unsigned i = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned shift = bid_y * size;
  if (i < size) {
    float gy = pgy[i + shift];
    atomic_add_float(pga + i + mba * shift, gy);
    atomic_add_float(pgb + i + mbb * shift, -gy);
  }
}
)EOS";

  ss << R"EOS(
kernel void multiply_bw_kernel(constant float *pa, constant float *pb, constant float *py, constant float *pgy,
                               constant unsigned *size_p, constant unsigned *mba_p, constant unsigned *mbb_p, global float *pga, global float *pgb) {
  unsigned size = size_p[0];
  unsigned mba = mba_p[0];
  unsigned mbb = mbb_p[0];
  unsigned i = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned shift = bid_y * size;
  if (i < size) {
    float gy = pgy[i + shift];
    unsigned a_ofs = i + mba * shift;
    unsigned b_ofs = i + mbb * shift;
    atomic_add_float(pga + a_ofs, gy * pb[b_ofs]);
    atomic_add_float(pgb + b_ofs, gy * pa[a_ofs]);
  }
}
)EOS";

  ss << R"EOS(
kernel void divide_bw_kernel(constant float *pa, constant float *pb, constant float *py, constant float *pgy,
                             constant unsigned *size_p, constant unsigned *mba_p, constant unsigned *mbb_p, global float *pga, global float *pgb) {
  unsigned size = size_p[0];
  unsigned mba = mba_p[0];
  unsigned mbb = mbb_p[0];
  unsigned i = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned shift = bid_y * size;
  if (i < size) {
    unsigned b_ofs = i + mbb * shift;
    unsigned y_ofs = i + shift;
    float k = pgy[y_ofs] / pb[b_ofs];
    atomic_add_float(pga + i + mba * shift, k);
    atomic_add_float(pgb + b_ofs, -k * py[y_ofs]);
  }
}
)EOS";

  ss << R"EOS(
kernel void transpose_fw_kernel(constant float *px, constant unsigned *rows_p, constant unsigned *cols_p, global float *py) {
  unsigned rows = rows_p[0];
  unsigned cols = cols_p[0];
  unsigned i = get_global_id(0);
  unsigned j = get_global_id(1);
  unsigned bid_z = get_group_id(2);
  unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}
)EOS";
  ss << R"EOS(
kernel void transpose_bw_kernel(constant float *py, constant unsigned *rows_p, constant unsigned *cols_p, global float *px) {
  unsigned rows = rows_p[0];
  unsigned cols = cols_p[0];
  unsigned i = get_global_id(0);
  unsigned j = get_global_id(1);
  unsigned bid_z = get_group_id(2);
  unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}
)EOS";

  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void sum_fw_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global float *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = 0;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) temp[tid] += px[i * skip];
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] += temp[tid + k]; \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = temp[0];
#undef GROUP_SIZE
}
)EOS";
  }
  ss << R"EOS(
inline float logsumexp2_fw_kernel(float a, float b) {
  return a > b
    ? a + log(1.f + exp(b - a))
    : b + log(1.f + exp(a - b));
}
)EOS";
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void logsumexp_fw_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global float *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = -1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    temp[tid] = logsumexp2_fw_kernel(temp[tid], px[i * skip]);
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = logsumexp2_fw_kernel(temp[tid], temp[tid + k]); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = temp[0];
#undef GROUP_SIZE
}
)EOS";
  }
  ss << R"EOS(
kernel void broadcast_fw_kernel(constant float *px, constant unsigned *skip1_p, constant unsigned *skip2_p,
                                constant unsigned *size_p, global float *py) {
  unsigned skip1 = skip1_p[0];
  unsigned skip2 = skip2_p[0];
  unsigned size = size_p[0];
  unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}
)EOS";
  ss << R"EOS(
kernel void batch_sum_fw_kernel(constant float *px, constant unsigned *size_p,
                                constant unsigned *batch_p, global float *py) {
  unsigned size = size_p[0];
  unsigned batch = batch_p[0];
  unsigned i = get_global_id(0);;
  if (i < size) {
    float temp = .0f;
    px += i;
    for (unsigned j = 0; j < batch; ++j, px += size) {
      temp += *px;
    }
    py[i] = temp;
  }
}
)EOS";
  ss << R"EOS(
kernel void inplace_multiply_const_kernel(constant float *k_p, constant unsigned *size_p, global float *px) {
  unsigned k = k_p[0];
  unsigned size = size_p[0];
  unsigned i = get_global_id(0);;
  if (i < size) px[i] *= k;
}
)EOS";
  ss << R"EOS(
kernel void inplace_add_kernel(constant float *px, constant unsigned *size_p,
                               constant unsigned *mbx_p, constant unsigned *mby_p, global float *py) {
  unsigned size = size_p[0];
  unsigned mbx = mbx_p[0];
  unsigned mby = mby_p[0];
  unsigned i = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, px[i + mbx * shift]);
}
)EOS";
  ss << R"EOS(
kernel void inplace_subtract_kernel(constant float *px, constant unsigned *size_p,
                                    constant unsigned *mbx_p, constant unsigned *mby_p, global float *py) {
  unsigned size = size_p[0];
  unsigned mbx = mbx_p[0];
  unsigned mby = mby_p[0];
  unsigned i = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, -px[i + mbx * shift]);
}
)EOS";

  return ss.str();
}

std::uint32_t num_platforms() {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  return all_platforms.size();
}

std::uint32_t num_devices(std::uint32_t platform_id) {\
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    THROW_ERROR("No platforms found. Check OpenCL installation!");
  }
  cl::Platform platform = all_platforms[platform_id];\
  std::vector<cl::Device> all_devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  return all_devices.size();
}

OpenCL::OpenCL(std::uint32_t platform_id, std::uint32_t device_id) {
  std::int32_t error = CL_SUCCESS;
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    THROW_ERROR("No platforms found. Check OpenCL installation!");
  }
  cl::Platform platform = all_platforms[platform_id];
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

  std::vector<cl::Device> all_devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size() == 0){
    THROW_ERROR("No devices found. Check OpenCL installation!");
  }
  dev_id_ = device_id;
  device_ = all_devices[dev_id_];
  context_ = cl::Context({device_});
  std::cout << "Using device: " << device_.getInfo<CL_DEVICE_NAME>() << std::endl;

  cl::Program::Sources sources;
  std::string kernel_code = kernel_code_generator();

  sources.push_back({kernel_code.c_str(), kernel_code.length()});
  cl::Program program(context_, sources);
  if (program.build({device_}, "-cl-std=CL2.0 -Werror") != CL_SUCCESS) {
    std::cerr << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
    THROW_ERROR("Error!");
  }
  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "argmax_kernel_" << (1 << i);
    argmax_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }
  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "argmin_kernel_" << (1 << i);
    argmin_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }

  set_identity_kernel_ = cl::Kernel(program, "set_identity_kernel", &error);
  pick_fw_kernel_ = cl::Kernel(program, "pick_fw_kernel", &error);
  slice_fw_kernel_ = cl::Kernel(program, "slice_fw_kernel", &error);
  concat_fw_kernel_ = cl::Kernel(program, "concat_fw_kernel", &error);
  pick_bw_kernel_ = cl::Kernel(program, "pick_bw_kernel", &error);
  slice_bw_kernel_ = cl::Kernel(program, "slice_bw_kernel", &error);

  negate_fw_kernel_ = cl::Kernel(program, "negate_fw_kernel", &error);
  sqrt_fw_kernel_ = cl::Kernel(program, "sqrt_fw_kernel", &error);
  exp_fw_kernel_ = cl::Kernel(program, "exp_fw_kernel", &error);
  log_fw_kernel_ = cl::Kernel(program, "log_fw_kernel", &error);
  tanh_fw_kernel_ = cl::Kernel(program, "tanh_fw_kernel", &error);
  sigmoid_fw_kernel_ = cl::Kernel(program, "sigmoid_fw_kernel", &error);
  softplus_fw_kernel_ = cl::Kernel(program, "softplus_fw_kernel", &error);
  sin_fw_kernel_ = cl::Kernel(program, "sin_fw_kernel", &error);
  cos_fw_kernel_ = cl::Kernel(program, "cos_fw_kernel", &error);
  tan_fw_kernel_ = cl::Kernel(program, "tan_fw_kernel", &error);
  transpose_fw_kernel_ = cl::Kernel(program, "transpose_fw_kernel", &error);

  sqrt_bw_kernel_ = cl::Kernel(program, "sqrt_bw_kernel", &error);
  exp_bw_kernel_ = cl::Kernel(program, "exp_bw_kernel", &error);
  log_bw_kernel_ = cl::Kernel(program, "log_bw_kernel", &error);
  tanh_bw_kernel_ = cl::Kernel(program, "tanh_bw_kernel", &error);
  sigmoid_bw_kernel_ = cl::Kernel(program, "sigmoid_bw_kernel", &error);
  softplus_bw_kernel_ = cl::Kernel(program, "softplus_bw_kernel", &error);
  sin_bw_kernel_ = cl::Kernel(program, "sin_bw_kernel", &error);
  cos_bw_kernel_ = cl::Kernel(program, "cos_bw_kernel", &error);
  tan_bw_kernel_ = cl::Kernel(program, "tan_bw_kernel", &error);
  transpose_bw_kernel_ = cl::Kernel(program, "transpose_bw_kernel", &error);

  add_const_fw_kernel_ = cl::Kernel(program, "add_const_fw_kernel", &error);
  subtract_const_r_fw_kernel_ = cl::Kernel(program, "subtract_const_r_fw_kernel", &error);
  subtract_const_l_fw_kernel_ = cl::Kernel(program, "subtract_const_l_fw_kernel", &error);
  multiply_const_fw_kernel_ = cl::Kernel(program, "multiply_const_fw_kernel", &error);
  divide_const_r_fw_kernel_ = cl::Kernel(program, "divide_const_r_fw_kernel", &error);
  divide_const_l_fw_kernel_ = cl::Kernel(program, "divide_const_l_fw_kernel", &error);
  prelu_fw_kernel_ = cl::Kernel(program, "prelu_fw_kernel", &error);
  elu_fw_kernel_ = cl::Kernel(program, "elu_fw_kernel", &error);

  add_const_bw_kernel_ = cl::Kernel(program, "add_const_bw_kernel", &error);
  subtract_const_r_bw_kernel_ = cl::Kernel(program, "subtract_const_r_bw_kernel", &error);
  subtract_const_l_bw_kernel_ = cl::Kernel(program, "subtract_const_l_bw_kernel", &error);
  multiply_const_bw_kernel_ = cl::Kernel(program, "multiply_const_bw_kernel", &error);
  divide_const_r_bw_kernel_ = cl::Kernel(program, "divide_const_r_bw_kernel", &error);
  divide_const_l_bw_kernel_ = cl::Kernel(program, "divide_const_l_bw_kernel", &error);
  prelu_bw_kernel_ = cl::Kernel(program, "prelu_bw_kernel", &error);
  elu_bw_kernel_ = cl::Kernel(program, "elu_bw_kernel", &error);

  add_scalar_fw_kernel_ = cl::Kernel(program, "add_scalar_fw_kernel", &error);
  subtract_scalar_r_fw_kernel_ = cl::Kernel(program, "subtract_scalar_r_fw_kernel", &error);
  subtract_scalar_l_fw_kernel_ = cl::Kernel(program, "subtract_scalar_l_fw_kernel", &error);
  multiply_scalar_fw_kernel_ = cl::Kernel(program, "multiply_scalar_fw_kernel", &error);
  divide_scalar_r_fw_kernel_ = cl::Kernel(program, "divide_scalar_r_fw_kernel", &error);
  divide_scalar_l_fw_kernel_ = cl::Kernel(program, "divide_scalar_l_fw_kernel", &error);

  add_fw_kernel_ = cl::Kernel(program, "add_fw_kernel", &error);
  subtract_fw_kernel_ = cl::Kernel(program, "subtract_fw_kernel", &error);
  multiply_fw_kernel_ = cl::Kernel(program, "multiply_fw_kernel", &error);
  divide_fw_kernel_ = cl::Kernel(program, "divide_fw_kernel", &error);
  matmul_fw_kernel_ = cl::Kernel(program, "matmul_fw_kernel", &error);

  add_bw_kernel_ = cl::Kernel(program, "add_bw_kernel", &error);
  subtract_bw_kernel_ = cl::Kernel(program, "subtract_bw_kernel", &error);
  multiply_bw_kernel_ = cl::Kernel(program, "multiply_bw_kernel", &error);
  divide_bw_kernel_ = cl::Kernel(program, "divide_bw_kernel", &error);
  matmul_bw_kernel_ = cl::Kernel(program, "matmul_bw_kernel", &error);

  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "sum_fw_kernel_" << (1 << i);
    sum_fw_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }
  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "logsumexp_fw_kernel_" << (1 << i);
    logsumexp_fw_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }

  cl::Kernel broadcast_fw_kernel_ = cl::Kernel(program, "broadcast_fw_kernel", &error);
  cl::Kernel batch_sum_fw_kernel_ = cl::Kernel(program, "batch_sum_fw_kernel", &error);

  cl::Kernel inplace_multiply_const_kernel_ = cl::Kernel(program, "inplace_multiply_const_kernel", &error);

  cl::Kernel inplace_add_kernel_ = cl::Kernel(program, "inplace_add_kernel", &error);
  cl::Kernel inplace_subtract_kernel_ = cl::Kernel(program, "inplace_subtract_kernel", &error);
}

OpenCL::~OpenCL() {
  // Nothing to do for now.
}

void OpenCL::dump_description() const {
  std::cerr << "Device " << this << ':' << std::endl;
  std::cerr << "  Type: OpenCL" << std::endl;
  // TODO
}

std::shared_ptr<void> OpenCL::new_handle(const Shape &shape) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t mem_size = sizeof(float) * shape.size();
  cl::Buffer *data = new cl::Buffer(context_,
            CL_MEM_READ_WRITE,
            mem_size,
            NULL, &error);
  if (error != CL_SUCCESS) {
    THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  return std::shared_ptr<void>(data, [](cl::Buffer *buffer){delete buffer;});
}

std::vector<float> OpenCL::tensor_to_vector_impl(const Tensor &x) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueReadBuffer(CDATA(x), CL_TRUE, 0,
            sizeof(cl_float) * num_elements, ret.data(), NULL, NULL);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmax_impl(const Tensor &x, std::uint32_t dim) {
  cl_int error = CL_SUCCESS;
  const Shape &shape = x.shape();
  std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = argmax_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  cl::Buffer py = cl::Buffer(context_,
      CL_MEM_WRITE_ONLY,
      sizeof(cl_uint) * r, NULL, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmax_kernel_[m].setArg(0, CDATA(x)); \
      argmax_kernel_[m].setArg(1, mem_s); \
      argmax_kernel_[m].setArg(2, mem_n); \
      argmax_kernel_[m].setArg(3, py); \
      queue.enqueueNDRangeKernel(argmax_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
  std::vector<std::uint32_t> ret(r);
  queue.enqueueReadBuffer(py, CL_TRUE, 0,
            sizeof(cl_uint) * r, ret.data(), NULL, NULL);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmin_impl(const Tensor &x, std::uint32_t dim) {
  cl_int error = CL_SUCCESS;
  const Shape &shape = x.shape();
  std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = argmin_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  cl::Buffer py = cl::Buffer(context_,
      CL_MEM_WRITE_ONLY,
      sizeof(cl_uint) * r, NULL, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmin_kernel_[m].setArg(0, CDATA(x)); \
      argmin_kernel_[m].setArg(1, mem_s); \
      argmin_kernel_[m].setArg(2, mem_n); \
      argmin_kernel_[m].setArg(3, py); \
      queue.enqueueNDRangeKernel(argmin_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
  std::vector<std::uint32_t> ret(r);
  queue.enqueueReadBuffer(py, CL_TRUE, 0,
            sizeof(cl_uint) * r, ret.data(), NULL, NULL);
  return ret;
}

void OpenCL::reset_tensor_impl(float k, Tensor &x) {
  std::int32_t error = CL_SUCCESS;
  const std::uint32_t size = x.shape().size();
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueFillBuffer<float>(CDATA(x), k, 0, sizeof(float) * size);
  queue.finish();
}

void OpenCL::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  std::int32_t error = CL_SUCCESS;
  const std::uint32_t size = x.shape().size();
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueWriteBuffer(CDATA(x), CL_TRUE, 0,
            sizeof(float) * size, values, NULL, NULL);
}

void OpenCL::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case Device::DEVICE_TYPE_CPU:
      reset_tensor_by_array(static_cast<const float *>((x).data()), y);
      break;
    case Device::DEVICE_TYPE_OPENCL:
      {
        std::int32_t error = CL_SUCCESS;
        const std::uint32_t size = x.shape().size();
        cl::CommandQueue queue(context_, device_, 0, &error);
        queue.enqueueCopyBuffer(CDATA(x), CDATA(y), 0, 0, sizeof(float) * size);
        queue.finish();
      }
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void OpenCL::identity_impl(Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t size = y.shape().size();
  std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t group_size = set_identity_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t num_blocks = GRID_SIZE(size, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  SET_ARG_HOST_SCALAR(set_identity_kernel_, 0, cl_uint, size)
  SET_ARG_HOST_SCALAR(set_identity_kernel_, 1, cl_uint, skip)
  set_identity_kernel_.setArg(2, CDATA(y));
  queue.enqueueNDRangeKernel(set_identity_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t wy = y.shape().lower_volume(dim);
  std::uint32_t wx = wy * x.shape()[dim];
  std::uint32_t sx = x.shape().has_batch() * x.shape().volume();
  std::uint32_t si = ids.size() > 1;
  std::uint32_t sy = y.shape().volume();
  const std::uint32_t group_size = pick_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(sy, group_size);
  const std::uint32_t bs = y.shape().batch();
  cl::CommandQueue queue(context_, device_, 0, &error);
  pick_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_VECTOR(pick_fw_kernel_, 1, cl_uint, ids)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 2, cl_uint, wx)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 3, cl_uint, wy)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 4, cl_uint, sx)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 5, cl_uint, si)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 6, cl_uint, sy)
  pick_fw_kernel_.setArg(7, CDATA(y));
  queue.enqueueNDRangeKernel(pick_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, bs), cl::NDRange(group_size, 1), NULL, NULL);
  queue.finish();
}

void OpenCL::slice_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t base = y.shape().lower_volume(dim);
  std::uint32_t shift = base * offset;
  std::uint32_t span = base * y.shape()[dim];
  std::uint32_t skip = base * x.shape()[dim];
  std::uint32_t size = y.shape().size();
  const std::uint32_t group_size = slice_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t num_blocks = GRID_SIZE(size, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  slice_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 1, cl_uint, shift)
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 2, cl_uint, span)
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 3, cl_uint, skip)
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 4, cl_uint, size)
  slice_fw_kernel_.setArg(5, CDATA(y));
  queue.enqueueNDRangeKernel(slice_fw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::concat_fw_impl(const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  std::uint32_t skip = base * y.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / skip;
  cl::CommandQueue queue(context_, device_, 0, &error);
  std::uint32_t offset = 0;
  const std::uint32_t group_size = concat_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  for (const Tensor *x : xs) {
    std::uint32_t span = base * x->shape()[dim];
    std::uint32_t x_size = span * repeat * x->shape().batch();
    std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = GRID_SIZE(y_size, group_size);
    concat_fw_kernel_.setArg(0, CDATA(*x));
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 1, cl_uint, span)
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 2, cl_uint, skip)
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 3, cl_uint, x_size)
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 4, cl_uint, y_size)
    concat_fw_kernel_.setArg(5, CDATA(y));
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 6, cl_uint, offset)
    queue.enqueueNDRangeKernel(concat_fw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL);
    offset += span;
  }
  queue.finish();
}

void OpenCL::pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &gx) {
  cl_int error = CL_SUCCESS;
  std::uint32_t wy = gy.shape().lower_volume(dim);
  std::uint32_t wx = wy * gx.shape()[dim];
  std::uint32_t sx = gx.shape().has_batch() * gx.shape().volume();
  std::uint32_t si = ids.size() > 1;
  std::uint32_t sy = gy.shape().volume();
  const std::uint32_t group_size = concat_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(sy, group_size);
  const std::uint32_t bs = gy.shape().batch();
  cl::CommandQueue queue(context_, device_, 0, &error);
  pick_bw_kernel_.setArg(0, CDATA(gy));
  SET_ARG_HOST_VECTOR(pick_bw_kernel_, 1, cl_uint, ids)
  SET_ARG_HOST_SCALAR(pick_bw_kernel_, 2, cl_uint, wx)
  SET_ARG_HOST_SCALAR(pick_bw_kernel_, 3, cl_uint, wy)
  SET_ARG_HOST_SCALAR(pick_bw_kernel_, 4, cl_uint, sx)
  SET_ARG_HOST_SCALAR(pick_bw_kernel_, 5, cl_uint, si)
  SET_ARG_HOST_SCALAR(pick_bw_kernel_, 6, cl_uint, sy)
  pick_bw_kernel_.setArg(7, CDATA(gx));
  queue.enqueueNDRangeKernel(pick_bw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, bs), cl::NDRange(group_size, 1), NULL, NULL);
  queue.finish();
}

void OpenCL::slice_bw_impl(const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  cl_int error = CL_SUCCESS;
  const Shape &sx = gx.shape();
  const Shape &sy = gy.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  std::uint32_t ox = base * offset;
  std::uint32_t wx = base * sx[dim];
  std::uint32_t wy = base * sy[dim];
  const std::uint32_t repeat = sx.volume() / wx;
  std::uint32_t nx = repeat * sx.batch();
  std::uint32_t ny = repeat * sy.batch();
  const std::uint32_t group_size = slice_bw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(wy * std::max(nx, ny), group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  slice_bw_kernel_.setArg(0, CDATA(gy));
  SET_ARG_HOST_SCALAR(slice_bw_kernel_, 1, cl_uint, wx)
  SET_ARG_HOST_SCALAR(slice_bw_kernel_, 2, cl_uint, wy)
  SET_ARG_HOST_SCALAR(slice_bw_kernel_, 3, cl_uint, nx)
  SET_ARG_HOST_SCALAR(slice_bw_kernel_, 4, cl_uint, ny)
  slice_bw_kernel_.setArg(5, CDATA(gx));
  SET_ARG_HOST_SCALAR(slice_bw_kernel_, 6, cl_uint, ox)
  queue.enqueueNDRangeKernel(slice_bw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

#define OPENCLDEV_FW_X(name) \
void OpenCL::name##_fw_impl(const Tensor &x, Tensor &y) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = x.shape().size(); \
  const std::uint32_t group_size = name##_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  const std::uint32_t num_blocks = GRID_SIZE(size, group_size); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_fw_kernel_.setArg(0, CDATA(x)); \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 1, cl_uint, size) \
  name##_fw_kernel_.setArg(2, CDATA(y)); \
  queue.enqueueNDRangeKernel(name##_fw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL); \
  queue.finish(); \
}

#define OPENCLDEV_BW_X(name) \
void OpenCL::name##_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = x.shape().size(); \
  const std::uint32_t group_size = name##_bw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  const std::uint32_t num_blocks = GRID_SIZE(size, group_size); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_bw_kernel_.setArg(0, CDATA(x)); \
  name##_bw_kernel_.setArg(1, CDATA(y)); \
  name##_bw_kernel_.setArg(2, CDATA(gy)); \
  SET_ARG_HOST_SCALAR(name##_bw_kernel_, 3, cl_uint, size) \
  name##_bw_kernel_.setArg(4, CDATA(gx)); \
  queue.enqueueNDRangeKernel(name##_bw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL); \
  queue.finish(); \
}

#define OPENCLDEV_FW_X_CONST(name) \
void OpenCL::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = x.shape().size(); \
  const std::uint32_t group_size = name##_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  std::uint32_t num_blocks = GRID_SIZE(size, group_size); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_fw_kernel_.setArg(0, CDATA(x)); \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 1, cl_float, k) \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 2, cl_uint, size) \
  name##_fw_kernel_.setArg(3, CDATA(y)); \
  queue.enqueueNDRangeKernel(name##_fw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL); \
  queue.finish(); \
}

#define OPENCLDEV_BW_X_CONST(name) \
void OpenCL::name##_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = x.shape().size(); \
  const std::uint32_t group_size = name##_bw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  std::uint32_t num_blocks = GRID_SIZE(size, group_size); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_bw_kernel_.setArg(0, CDATA(x)); \
  name##_bw_kernel_.setArg(1, CDATA(y)); \
  name##_bw_kernel_.setArg(2, CDATA(gy)); \
  SET_ARG_HOST_SCALAR(name##_bw_kernel_, 3, cl_float, k) \
  SET_ARG_HOST_SCALAR(name##_bw_kernel_, 4, cl_uint, size) \
  name##_bw_kernel_.setArg(5, CDATA(gx)); \
  queue.enqueueNDRangeKernel(name##_bw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL); \
  queue.finish(); \
}

#define OPENCLDEV_FW_X_SCALAR(name) \
void OpenCL::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = y.shape().volume(); \
  const std::uint32_t group_size = name##_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  const std::uint32_t g1 = GRID_SIZE(size, group_size); \
  const std::uint32_t g2 = y.shape().batch(); \
  std::uint32_t mbx = x.shape().has_batch(); \
  std::uint32_t mbk = k.shape().has_batch(); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_fw_kernel_.setArg(0, CDATA(x)); \
  name##_fw_kernel_.setArg(1, CDATA(k)); \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 2, cl_uint, size) \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 3, cl_uint, mbx) \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 4, cl_uint, mbk) \
  name##_fw_kernel_.setArg(5, CDATA(y)); \
  queue.enqueueNDRangeKernel(name##_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, g2, 1), cl::NDRange(group_size, 1, 1), NULL, NULL); \
  queue.finish(); \
}

#define OPENCLDEV_FW_AB(name) \
void OpenCL::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = y.shape().volume(); \
  const std::uint32_t group_size = name##_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  const std::uint32_t g1 = GRID_SIZE(size, group_size); \
  const std::uint32_t g2 = y.shape().batch(); \
  std::uint32_t mba = a.shape().has_batch(); \
  std::uint32_t mbb = b.shape().has_batch(); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_fw_kernel_.setArg(0, CDATA(a)); \
  name##_fw_kernel_.setArg(1, CDATA(b)); \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 2, cl_uint, size) \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 3, cl_uint, mba) \
  SET_ARG_HOST_SCALAR(name##_fw_kernel_, 4, cl_uint, mbb) \
  name##_fw_kernel_.setArg(5, CDATA(y)); \
  queue.enqueueNDRangeKernel(name##_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, g2, 1), cl::NDRange(group_size, 1, 1), NULL, NULL); \
  queue.finish(); \
}

#define OPENCLDEV_BW_AB(name) \
void OpenCL::name##_bw_impl(const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, Tensor &ga, Tensor &gb) { \
  cl_int error = CL_SUCCESS; \
  std::uint32_t size = y.shape().volume(); \
  const std::uint32_t group_size = name##_bw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_); \
  const std::uint32_t g1 = GRID_SIZE(size, group_size); \
  const std::uint32_t g2 = y.shape().batch(); \
  std::uint32_t mba = a.shape().has_batch(); \
  std::uint32_t mbb = b.shape().has_batch(); \
  cl::CommandQueue queue(context_, device_, 0, &error); \
  name##_bw_kernel_.setArg(0, CDATA(a)); \
  name##_bw_kernel_.setArg(1, CDATA(b)); \
  name##_bw_kernel_.setArg(2, CDATA(y)); \
  name##_bw_kernel_.setArg(3, CDATA(gy)); \
  SET_ARG_HOST_SCALAR(name##_bw_kernel_, 4, cl_uint, size) \
  SET_ARG_HOST_SCALAR(name##_bw_kernel_, 5, cl_uint, mba) \
  SET_ARG_HOST_SCALAR(name##_bw_kernel_, 6, cl_uint, mbb) \
  name##_bw_kernel_.setArg(7, CDATA(ga)); \
  name##_bw_kernel_.setArg(8, CDATA(gb)); \
  queue.enqueueNDRangeKernel(name##_bw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, g2, 1), cl::NDRange(group_size, 1, 1), NULL, NULL); \
  queue.finish(); \
}

OPENCLDEV_FW_X(negate);
OPENCLDEV_FW_X(sqrt);
OPENCLDEV_FW_X(exp);
OPENCLDEV_FW_X(log);
OPENCLDEV_FW_X(tanh);
OPENCLDEV_FW_X(sigmoid);
OPENCLDEV_FW_X(softplus);
OPENCLDEV_FW_X(sin);
OPENCLDEV_FW_X(cos);
OPENCLDEV_FW_X(tan);

OPENCLDEV_BW_X(sqrt);
OPENCLDEV_BW_X(exp);
OPENCLDEV_BW_X(log);
OPENCLDEV_BW_X(tanh);
OPENCLDEV_BW_X(sigmoid);
OPENCLDEV_BW_X(softplus);
OPENCLDEV_BW_X(sin);
OPENCLDEV_BW_X(cos);
OPENCLDEV_BW_X(tan);

OPENCLDEV_FW_X_CONST(add_const);
OPENCLDEV_FW_X_CONST(subtract_const_r);
OPENCLDEV_FW_X_CONST(subtract_const_l);
OPENCLDEV_FW_X_CONST(multiply_const);
OPENCLDEV_FW_X_CONST(divide_const_r);
OPENCLDEV_FW_X_CONST(divide_const_l);
OPENCLDEV_FW_X_CONST(prelu);
OPENCLDEV_FW_X_CONST(elu);

OPENCLDEV_BW_X_CONST(add_const);
OPENCLDEV_BW_X_CONST(subtract_const_r);
OPENCLDEV_BW_X_CONST(subtract_const_l);
OPENCLDEV_BW_X_CONST(multiply_const);
OPENCLDEV_BW_X_CONST(divide_const_r);
OPENCLDEV_BW_X_CONST(divide_const_l);
OPENCLDEV_BW_X_CONST(prelu);
OPENCLDEV_BW_X_CONST(elu);

OPENCLDEV_FW_X_SCALAR(add_scalar);
OPENCLDEV_FW_X_SCALAR(subtract_scalar_r);
OPENCLDEV_FW_X_SCALAR(subtract_scalar_l);
OPENCLDEV_FW_X_SCALAR(multiply_scalar);
OPENCLDEV_FW_X_SCALAR(divide_scalar_r);
OPENCLDEV_FW_X_SCALAR(divide_scalar_l);

OPENCLDEV_FW_AB(add);
OPENCLDEV_FW_AB(subtract);
OPENCLDEV_FW_AB(multiply);
OPENCLDEV_FW_AB(divide);

OPENCLDEV_BW_AB(add);
OPENCLDEV_BW_AB(subtract);
OPENCLDEV_BW_AB(multiply);
OPENCLDEV_BW_AB(divide);

#undef OPENCLDEV_FW_X
#undef OPENCLDEV_BW_X
#undef OPENCLDEV_FW_X_CONST
#undef OPENCLDEV_BW_X_CONST
#undef OPENCLDEV_FW_X_SCALAR
#undef OPENCLDEV_FW_AB
#undef OPENCLDEV_BW_AB

void OpenCL::transpose_fw_impl(const Tensor &x, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t rows = x.shape()[0];
  std::uint32_t cols = x.shape()[1];
  const std::uint32_t bs = x.shape().batch();
  std::uint32_t group_size_y = transpose_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  std::uint32_t group_size_x = 1;
  while (group_size_x < group_size_y) {
    group_size_x <<= 1;
    group_size_y >>= 1;
  }
  const std::uint32_t g1 = GRID_SIZE(rows, group_size_x);
  const std::uint32_t g2 = GRID_SIZE(cols, group_size_y);
  cl::CommandQueue queue(context_, device_, 0, &error);
  transpose_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(transpose_fw_kernel_, 1, cl_uint, rows)
  SET_ARG_HOST_SCALAR(transpose_fw_kernel_, 2, cl_uint, cols)
  transpose_fw_kernel_.setArg(3, CDATA(y));
  queue.enqueueNDRangeKernel(transpose_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size_x, g2 * group_size_y, bs), cl::NDRange(group_size_x, group_size_y, 1), NULL, NULL);
  queue.finish();
}

void OpenCL::transpose_bw_impl(const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  cl_int error = CL_SUCCESS;
  std::uint32_t rows = gx.shape()[0];
  std::uint32_t cols = gx.shape()[1];
  const std::uint32_t bs = gx.shape().batch();
  std::uint32_t group_size_y = transpose_bw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  std::uint32_t group_size_x = 1;
  while (group_size_x < group_size_y) {
    group_size_x <<= 1;
    group_size_y >>= 1;
  }
  const std::uint32_t g1 = GRID_SIZE(rows, group_size_x);
  const std::uint32_t g2 = GRID_SIZE(cols, group_size_y);
  cl::CommandQueue queue(context_, device_, 0, &error);
  transpose_bw_kernel_.setArg(0, CDATA(gy));
  SET_ARG_HOST_SCALAR(transpose_bw_kernel_, 1, cl_uint, rows)
  SET_ARG_HOST_SCALAR(transpose_bw_kernel_, 2, cl_uint, cols)
  transpose_bw_kernel_.setArg(3, CDATA(gx));
  queue.enqueueNDRangeKernel(transpose_bw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size_x, g2 * group_size_y, bs), cl::NDRange(group_size_x, group_size_y, 1), NULL, NULL);
  queue.finish();
}

void OpenCL::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = sum_fw_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      sum_fw_kernel_[m].setArg(0, CDATA(x)); \
      sum_fw_kernel_[m].setArg(1, mem_s); \
      sum_fw_kernel_[m].setArg(2, mem_n); \
      sum_fw_kernel_[m].setArg(3, CDATA(y)); \
      queue.enqueueNDRangeKernel(sum_fw_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = logsumexp_fw_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      logsumexp_fw_kernel_[m].setArg(0, CDATA(x)); \
      logsumexp_fw_kernel_[m].setArg(1, mem_s); \
      logsumexp_fw_kernel_[m].setArg(2, mem_n); \
      logsumexp_fw_kernel_[m].setArg(3, CDATA(y)); \
      queue.enqueueNDRangeKernel(logsumexp_fw_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::broadcast_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t skip1 = y.shape().lower_volume(dim);
  std::uint32_t skip2 = skip1 * size;
  std::uint32_t total = y.shape().size();
  const std::uint32_t group_size = broadcast_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(total, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  broadcast_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(broadcast_fw_kernel_, 1, cl_uint, skip1)
  SET_ARG_HOST_SCALAR(broadcast_fw_kernel_, 2, cl_uint, skip2)
  SET_ARG_HOST_SCALAR(broadcast_fw_kernel_, 3, cl_uint, total)
  broadcast_fw_kernel_.setArg(4, CDATA(y));
  queue.enqueueNDRangeKernel(broadcast_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t size = y.shape().size();
  std::uint32_t batch = x.shape().batch();
  const std::uint32_t group_size = batch_sum_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(size, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  batch_sum_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(batch_sum_fw_kernel_, 1, cl_uint, size)
  SET_ARG_HOST_SCALAR(batch_sum_fw_kernel_, 2, cl_uint, batch)
  batch_sum_fw_kernel_.setArg(3, CDATA(y));
  queue.enqueueNDRangeKernel(batch_sum_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::inplace_multiply_const_impl(float k, Tensor &x) {
  cl_int error = CL_SUCCESS;
  std::uint32_t size = x.shape().size();
  const std::uint32_t group_size = inplace_multiply_const_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(size, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  SET_ARG_HOST_SCALAR(inplace_multiply_const_kernel_, 0, cl_float, k)
  SET_ARG_HOST_SCALAR(inplace_multiply_const_kernel_, 1, cl_uint, size)
  inplace_multiply_const_kernel_.setArg(2, CDATA(x));
  queue.enqueueNDRangeKernel(inplace_multiply_const_kernel_, cl::NullRange, cl::NDRange(g1 * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::inplace_add_impl(const Tensor &x, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t size = y.shape().size();
  std::uint32_t mbx = x.shape().has_batch();
  std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t group_size = inplace_add_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(size, group_size);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  cl::CommandQueue queue(context_, device_, 0, &error);
  inplace_add_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(inplace_add_kernel_, 1, cl_uint, size)
  SET_ARG_HOST_SCALAR(inplace_add_kernel_, 2, cl_uint, mbx)
  SET_ARG_HOST_SCALAR(inplace_add_kernel_, 3, cl_uint, mby)
  inplace_add_kernel_.setArg(4, CDATA(y));
  queue.enqueueNDRangeKernel(inplace_add_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, bs, 1), cl::NDRange(group_size, 1, 1), NULL, NULL);
  queue.finish();
}

void OpenCL::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t size = y.shape().size();
  std::uint32_t mbx = x.shape().has_batch();
  std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t group_size = inplace_subtract_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(size, group_size);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  cl::CommandQueue queue(context_, device_, 0, &error);
  inplace_subtract_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(inplace_subtract_kernel_, 1, cl_uint, size)
  SET_ARG_HOST_SCALAR(inplace_subtract_kernel_, 2, cl_uint, mbx)
  SET_ARG_HOST_SCALAR(inplace_subtract_kernel_, 3, cl_uint, mby)
  inplace_subtract_kernel_.setArg(4, CDATA(y));
  queue.enqueueNDRangeKernel(inplace_subtract_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, bs, 1), cl::NDRange(group_size, 1, 1), NULL, NULL);
  queue.finish();
}

}
}
