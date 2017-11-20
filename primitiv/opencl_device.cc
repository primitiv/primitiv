#include <config.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/cl2.hpp>
#include <clBLAS.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <primitiv/error.h>
#include <primitiv/opencl_device.h>

namespace {

/**
 * Creates a readonly buffer using given vector.
 * @param context OpenCL context object.
 * @param data Target vector.
 * @return New OpenCL buffer.
 */
template<typename T>
cl::Buffer make_readonly_buffer(
    const cl::Context &context, const std::vector<T> &data) {
  return cl::Buffer(
      context,
      CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
      sizeof(T) * data.size(),
      const_cast<T *>(data.data()));
}

/**
 * Obtains mutable cl::Buffer from Tensor.
 * @param ptr Target Tensor object.
 * @return cl::Buffer object which the tensor object holds.
 */
cl::Buffer &get_buffer(primitiv::Tensor &ptr) {
  return *static_cast<cl::Buffer *>(ptr.data());
}

/**
 * Obtains immutable cl::Buffer from Tensor.
 * @param ptr Target Tensor object.
 * @return cl::Buffer object which the tensor object holds.
 */
const cl::Buffer &get_buffer(const primitiv::Tensor &ptr) {
  return *static_cast<const cl::Buffer *>(ptr.data());
}

/**
 * Obtains the number of blocks in one parallel operation.
 * @param size Total number of threads.
 * @param num_threads Number of threads in one block.
 */
std::uint32_t calc_num_blocks(std::uint32_t size, std::uint32_t num_threads) {
  return (size + num_threads - 1) / num_threads;
}

/**
 * Generates source code of all kernel functions.
 * @return Source code of kernel functions.
 */
std::string generate_kernels() {
  std::ostringstream ss;
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void argmax_kernel_" << group_size << R"EOS((
    const global float *px, const unsigned skip,
    const unsigned n, global unsigned *py) {
#define GROUP_SIZE )EOS" << group_size << R"EOS(
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float max_val[GROUP_SIZE];
  local unsigned argmax_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = -1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    const float val = px[i * skip];
    if (val > max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] > max_val[tid]) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
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
"kernel void argmin_kernel_" << group_size << R"EOS((
    const global float *px, const unsigned skip,
    const unsigned n, global unsigned *py) {
#define GROUP_SIZE )EOS" << group_size << R"EOS(
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float min_val[GROUP_SIZE];
  local unsigned argmin_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  min_val[tid] = 1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    const float val = px[i * skip];
    if (val < min_val[tid]) {
      min_val[tid] = val;
      argmin_val[tid] = i;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (min_val[tid + k] < min_val[tid]) { \
        min_val[tid] = min_val[tid + k]; \
        argmin_val[tid] = argmin_val[tid + k]; \
      } \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
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
  if (tid == 0) py[bid] = argmin_val[0];
#undef GROUP_SIZE
}
)EOS";
  }
  ss << R"EOS(
kernel void set_identity_kernel(
    const unsigned size, const unsigned skip, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = !(i % skip);
}
)EOS";
  ss << R"EOS(
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
)EOS";
  ss << R"EOS(
kernel void slice_fw_kernel(
    const global float *px, const unsigned shift, const unsigned span,
    const unsigned skip, const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[(i / span) * skip + (i % span) + shift];
}
)EOS";
  ss << R"EOS(
kernel void concat_fw_kernel(
    const global float *px, const unsigned span, const unsigned skip,
    const unsigned x_size, const unsigned y_size,
    global float *py, const unsigned shift) {
  const unsigned i = get_global_id(0);
  if (i < y_size) py[(i / span) * skip + (i % span) + shift] = px[i % x_size];
}
)EOS";
  ss << R"EOS(
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
)EOS";
  ss << R"EOS(
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
)EOS";
  ss << R"EOS(
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
)EOS";

#define OPENCLDEV_KERNEL_FW_X(name, op) \
  ss << "kernel void " << name << "_fw_kernel(" \
        "   const global float *px, const unsigned size, global float *py) {" \
        " const unsigned i = get_global_id(0);" \
        " if (i < size) py[i] = (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_BW_X(name, op) \
  ss << "kernel void " << name << "_bw_kernel(" \
        "   const global float *px, const global float *py, const global float *pgy," \
        "   const unsigned size, global float *pgx) {" \
        " const unsigned i = get_global_id(0);" \
        " if (i < size) pgx[i] += (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_X_CONST(name, op) \
  ss << "kernel void " << name << "_fw_kernel(" \
        "   const global float *px, const float k," \
        "   const unsigned size, global float *py) {" \
        " const unsigned i = get_global_id(0);" \
        " if (i < size) py[i] = (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_BW_X_CONST(name, op) \
  ss << "kernel void " << name << "_bw_kernel(" \
        "   const global float *px, const global float *py, const global float *pgy," \
        "   const float k, const unsigned size, global float *pgx) {" \
        " const unsigned i = get_global_id(0);" \
        " if (i < size) pgx[i] += (" << op << ");" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_X_SCALAR_R_INFIX(name, op) \
  ss << "kernel void " << name << "_fw_kernel(" \
        "   const global float *px, const global float *pk, const unsigned size," \
        "   const unsigned mbx, const unsigned mbk, global float *py) {" \
        " const unsigned i = get_global_id(0);" \
        " const unsigned bid_y = get_group_id(1);" \
        " const unsigned shift = bid_y * size;" \
        " if (i < size) {" \
        "   py[i + shift] = px[i + mbx * shift] " << op << " pk[mbk * bid_y];" \
        " }" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_X_SCALAR_L_INFIX(name, op) \
  ss << "kernel void " << name << "_fw_kernel(" \
        "   const global float *px, const global float *pk, const unsigned size," \
        "   const unsigned mbx, const unsigned mbk, global float *py) {" \
        " const unsigned i = get_global_id(0);" \
        " const unsigned bid_y = get_group_id(1);" \
        " const unsigned shift = bid_y * size;" \
        " if (i < size) {" \
        "   py[i + shift] = pk[mbk * bid_y] " << op << " px[i + mbx * shift];" \
        " }" \
        "}\n";

#define OPENCLDEV_KERNEL_FW_AB_INFIX(name, op) \
  ss << "kernel void " << name << "_fw_kernel(" \
        "   const global float *pa, const global float *pb, const unsigned size," \
        "   const unsigned mba, const unsigned mbb, global float *py) {" \
        " const unsigned i = get_global_id(0);" \
        " const unsigned bid_y = get_group_id(1);" \
        " const unsigned shift = bid_y * size;" \
        " if (i < size) {" \
        "   py[i + shift] = pa[i + mba * shift] " << op << \
        "       pb[i + mbb * shift];" \
        " }" \
        "}\n";

OPENCLDEV_KERNEL_FW_X("negate", "-px[i]");
OPENCLDEV_KERNEL_FW_X("sqrt", "sqrt(px[i])");
OPENCLDEV_KERNEL_FW_X("exp", "exp(px[i])");
OPENCLDEV_KERNEL_FW_X("log", "log(px[i])");
OPENCLDEV_KERNEL_FW_X("tanh", "tanh(px[i])");
OPENCLDEV_KERNEL_FW_X("sigmoid", ".5f + .5f * tanh(.5f * px[i])");
OPENCLDEV_KERNEL_FW_X(
    "softplus", "max(px[i], .0f) + log(1.f + exp(-fabs(px[i])))");
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
OPENCLDEV_KERNEL_FW_X_CONST(
    "elu", "max(px[i], .0f) + k * (exp(min(px[i], .0f)) - 1.0f)");

OPENCLDEV_KERNEL_BW_X_CONST("add_const", "pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("subtract_const_r", "pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("subtract_const_l", "-pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("multiply_const", "k * pgy[i]");
OPENCLDEV_KERNEL_BW_X_CONST("divide_const_r", "pgy[i] / k");
OPENCLDEV_KERNEL_BW_X_CONST("divide_const_l", "-py[i] * pgy[i] / px[i]");
OPENCLDEV_KERNEL_BW_X_CONST(
    "prelu", "pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f))");
OPENCLDEV_KERNEL_BW_X_CONST(
    "elu", "pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f))");

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
)EOS";

  ss << R"EOS(
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
)EOS";

  ss << R"EOS(
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
)EOS";

  ss << R"EOS(
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
)EOS";

  ss << R"EOS(
kernel void transpose_fw_kernel(
    const global float *px, unsigned rows, unsigned cols, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}
)EOS";
  ss << R"EOS(
kernel void transpose_bw_kernel(
    const global float *py, const unsigned rows, const unsigned cols,
    global float *px) {
  const unsigned i = get_global_id(0);
  const unsigned j = get_global_id(1);
  const unsigned bid_z = get_group_id(2);
  const unsigned ofs = bid_z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}
)EOS";

  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void sum_fw_kernel_" << group_size << R"EOS((
    const global float *px, const unsigned skip, const unsigned n,
    global float *py) {
#define GROUP_SIZE )EOS" << group_size << R"EOS(
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = 0;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) temp[tid] += px[i * skip];
  barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] += temp[tid + k]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
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
"kernel void logsumexp_fw_kernel_" << group_size << R"EOS((
    const global float *px, const unsigned skip, const unsigned n,
    global float *py) {
#define GROUP_SIZE )EOS" << group_size << R"EOS(
  const unsigned bid = get_group_id(0);
  const unsigned tid = get_local_id(0);
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = -1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    temp[tid] = logsumexp2_fw_kernel(temp[tid], px[i * skip]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = logsumexp2_fw_kernel(temp[tid], temp[tid + k]); \
    barrier(CLK_LOCAL_MEM_FENCE); \
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
kernel void broadcast_fw_kernel(
    const global float *px, const unsigned skip1, const unsigned skip2,
    const unsigned size, global float *py) {
  const unsigned i = get_global_id(0);
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}
)EOS";
  ss << R"EOS(
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
)EOS";
  ss << R"EOS(
kernel void inplace_multiply_const_kernel(
    const float k, const unsigned size, global float *px) {
  const unsigned i = get_global_id(0);
  if (i < size) px[i] *= k;
}
)EOS";
  ss << R"EOS(
kernel void inplace_add_kernel(
    const global float *px, const unsigned size,
    const unsigned mbx, const unsigned mby, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, px[i + mbx * shift]);
}
)EOS";
  ss << R"EOS(
kernel void inplace_subtract_kernel(
    const global float *px, const unsigned size,
    const unsigned mbx, const unsigned mby, global float *py) {
  const unsigned i = get_global_id(0);
  const unsigned bid_y = get_group_id(1);
  const unsigned shift = bid_y * size;
  if (i < size) atomic_add_float(py + i + mby * shift, -px[i + mbx * shift]);
}
)EOS";

  return ss.str();
}

/**
 * Returns the list of available platforms.
 * @return List of available cl::Platform.
 */
std::vector<cl::Platform> get_all_platforms() {
  std::vector<cl::Platform> ret;
  cl::Platform::get(&ret);
  return ret;
}

/**
 * Returns the list of available devices on the specified platform.
 * @param platform_id Platform ID.
 * @return List of available cl::Device.
 */
std::vector<cl::Device> get_all_devices(std::uint32_t platform_id) {
  const auto all_pfs = ::get_all_platforms();
  if (platform_id >= all_pfs.size()) {
    THROW_ERROR("Invalid platform ID: " << platform_id);
  }
  std::vector<cl::Device> ret;
  all_pfs[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &ret);
  return ret;
}

/**
 * Returns the cl::Device corresponding to the specified IDs.
 * @param platform_id Platform ID.
 * @param device_id Device ID.
 * @return Corresponding cl::Device object.
 */
cl::Device get_device(std::uint32_t platform_id, std::uint32_t device_id) {
  const auto all_devs = ::get_all_devices(platform_id);
  if (device_id >= all_devs.size()) {
    THROW_ERROR(
        "Invalid device ID: " << device_id
        << " (on the platform " << platform_id << ")");
  }
  return all_devs[device_id];
}

}  // namespace

namespace primitiv {
namespace devices {

std::uint32_t OpenCL::num_platforms() {
  return ::get_all_platforms().size();
}

std::uint32_t OpenCL::num_devices(std::uint32_t platform_id) {
  return ::get_all_devices(platform_id).size();
}

void OpenCL::assert_support(
    std::uint32_t platform_id, std::uint32_t device_id) {
  const cl::Device dev = ::get_device(platform_id, device_id);

  // Checks whether the device is globally available.
  if (!dev.getInfo<CL_DEVICE_AVAILABLE>()) {
    THROW_ERROR(
        "OpenCL Device " << device_id << " on the platform " << platform_id
        << " is not available (CL_DEVICE_AVAILABLE == false).");
  }

  // Checks other minimum requirements.
#define CHECK_REQUIREMENT(name, value) \
  { \
    const auto actual = dev.getInfo<name>(); \
    if (actual < (value)) { \
      THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << ", " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  }
#define CHECK_REQUIREMENT_VECTOR(name, index, value) \
  { \
    const auto actual = dev.getInfo<name>()[index]; \
    if (actual < (value)) { \
      THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << "[" << #index << "], " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  } \

  CHECK_REQUIREMENT(CL_DEVICE_GLOBAL_MEM_SIZE, 1ull * (1ull << 30));
  CHECK_REQUIREMENT(CL_DEVICE_LOCAL_MEM_SIZE, 16ull * (1ull << 10));
  CHECK_REQUIREMENT(CL_DEVICE_MAX_WORK_GROUP_SIZE, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 1, 16);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 2, 1);
  // NOTE(odashi): OpenCL does not support explicit grid sizes.

#undef CHECK_REQUIREMENT
#undef CHECK_REQUIREMENT_VECTOR
}

void OpenCL::initialize() {
  assert_support(pf_id_, dev_id_);

  device_ = ::get_device(pf_id_, dev_id_);
  context_ = cl::Context({device_});
  cmd_queue_ = cl::CommandQueue(context_, device_, 0);
  cl::Program program(context_, ::generate_kernels(), true);

  auto get_kernel_group_size = [&](const cl::Kernel &kernel) {
    return kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  };

#define CONFIGURE_KERNEL(name) \
  { \
    name##_kernel_ = cl::Kernel(program, #name "_kernel"); \
    name##_kernel_group_size_ = get_kernel_group_size(name##_kernel_); \
  }

#define CONFIGURE_KERNEL_LIST(name) \
  { \
    for (std::uint32_t i = 0; i <= 10; ++i) { \
      std::ostringstream ss; \
      ss << #name "_kernel_" << (1 << i); \
      name##_kernel_[i] = cl::Kernel(program, ss.str().c_str()); \
    } \
    name##_kernel_group_size_ = get_kernel_group_size(name##_kernel_[0]); \
  }

  CONFIGURE_KERNEL_LIST(argmax);
  CONFIGURE_KERNEL_LIST(argmin);

  CONFIGURE_KERNEL(set_identity);

  CONFIGURE_KERNEL(pick_fw);
  CONFIGURE_KERNEL(slice_fw);
  CONFIGURE_KERNEL(concat_fw);

  CONFIGURE_KERNEL(pick_bw);
  CONFIGURE_KERNEL(slice_bw);

  CONFIGURE_KERNEL(negate_fw);
  CONFIGURE_KERNEL(sqrt_fw);
  CONFIGURE_KERNEL(exp_fw);
  CONFIGURE_KERNEL(log_fw);
  CONFIGURE_KERNEL(tanh_fw);
  CONFIGURE_KERNEL(sigmoid_fw);
  CONFIGURE_KERNEL(softplus_fw);
  CONFIGURE_KERNEL(sin_fw);
  CONFIGURE_KERNEL(cos_fw);
  CONFIGURE_KERNEL(tan_fw);

  CONFIGURE_KERNEL(sqrt_bw);
  CONFIGURE_KERNEL(exp_bw);
  CONFIGURE_KERNEL(log_bw);
  CONFIGURE_KERNEL(tanh_bw);
  CONFIGURE_KERNEL(sigmoid_bw);
  CONFIGURE_KERNEL(softplus_bw);
  CONFIGURE_KERNEL(sin_bw);
  CONFIGURE_KERNEL(cos_bw);
  CONFIGURE_KERNEL(tan_bw);

  CONFIGURE_KERNEL(transpose_fw);
  CONFIGURE_KERNEL(transpose_bw);

  // helper to find two sizes (x, y) that satisfy:
  // 1. x * y <= size
  // 2. x / y == 1 or 2
  auto calc_dim2_sizes = [](
      std::uint32_t size, std::uint32_t &x, std::uint32_t &y) {
    x = y = 1;
    bool p = true;
    while ((x * y) << 1 <= size) {
      (p ? x : y) <<= 1;
      p = !p;
    }
  };

  calc_dim2_sizes(
      transpose_fw_kernel_group_size_,
      transpose_fw_kernel_group_size_x_, transpose_fw_kernel_group_size_y_);
  calc_dim2_sizes(
      transpose_bw_kernel_group_size_,
      transpose_bw_kernel_group_size_x_, transpose_bw_kernel_group_size_y_);

  CONFIGURE_KERNEL(add_const_fw);
  CONFIGURE_KERNEL(subtract_const_r_fw);
  CONFIGURE_KERNEL(subtract_const_l_fw);
  CONFIGURE_KERNEL(multiply_const_fw);
  CONFIGURE_KERNEL(divide_const_r_fw);
  CONFIGURE_KERNEL(divide_const_l_fw);

  CONFIGURE_KERNEL(add_const_bw);
  CONFIGURE_KERNEL(subtract_const_r_bw);
  CONFIGURE_KERNEL(subtract_const_l_bw);
  CONFIGURE_KERNEL(multiply_const_bw);
  CONFIGURE_KERNEL(divide_const_r_bw);
  CONFIGURE_KERNEL(divide_const_l_bw);

  CONFIGURE_KERNEL(prelu_fw);
  CONFIGURE_KERNEL(elu_fw);

  CONFIGURE_KERNEL(prelu_bw);
  CONFIGURE_KERNEL(elu_bw);

  CONFIGURE_KERNEL(add_scalar_fw);
  CONFIGURE_KERNEL(subtract_scalar_r_fw);
  CONFIGURE_KERNEL(subtract_scalar_l_fw);
  CONFIGURE_KERNEL(multiply_scalar_fw);
  CONFIGURE_KERNEL(divide_scalar_r_fw);
  CONFIGURE_KERNEL(divide_scalar_l_fw);

  CONFIGURE_KERNEL(add_fw);
  CONFIGURE_KERNEL(subtract_fw);
  CONFIGURE_KERNEL(multiply_fw);
  CONFIGURE_KERNEL(divide_fw);

  CONFIGURE_KERNEL(add_bw);
  CONFIGURE_KERNEL(subtract_bw);
  CONFIGURE_KERNEL(multiply_bw);
  CONFIGURE_KERNEL(divide_bw);

  CONFIGURE_KERNEL_LIST(sum_fw);
  CONFIGURE_KERNEL_LIST(logsumexp_fw);

  CONFIGURE_KERNEL(broadcast_fw);
  CONFIGURE_KERNEL(batch_sum_fw);

  CONFIGURE_KERNEL(inplace_multiply_const);
  CONFIGURE_KERNEL(inplace_add);
  CONFIGURE_KERNEL(inplace_subtract);

#undef CONFIGURE_KERNEL
#undef CONFIGURE_KERNEL_LIST
}

OpenCL::OpenCL(std::uint32_t platform_id, std::uint32_t device_id)
: pf_id_(platform_id)
, dev_id_(device_id)
, randomizer_(std::random_device()()) {
  initialize();
}

OpenCL::OpenCL(
    std::uint32_t platform_id, std::uint32_t device_id, std::uint32_t rng_seed)
: pf_id_(platform_id)
, dev_id_(device_id)
, randomizer_(rng_seed) {
  initialize();
}

void OpenCL::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: OpenCL" << std::endl;

  std::cerr << "  Platform ID: " << pf_id_ << std::endl;
  std::cerr << "  Device ID: " << dev_id_ << std::endl;
  std::cerr << "    Vendor ............ "
            << device_.getInfo<CL_DEVICE_VENDOR>() << std::endl;
  std::cerr << "    Name .............. "
            << device_.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cerr << "    Global memory ..... "
            << device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
  std::cerr << "    Local memory ...... "
            << device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
  std::cerr << "    Work group size ... "
            << device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  std::cerr << "    Work item size .... ";
  const auto sizes = device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << sizes[i];
  }
  std::cerr << std::endl;
}

std::shared_ptr<void> OpenCL::new_handle(const Shape &shape) {
  const std::size_t mem_size = sizeof(float) * shape.size();
  cl::Buffer *data = new cl::Buffer(
      context_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, mem_size, NULL);
  return std::shared_ptr<void>(data, [&](void *ptr) {
      // NOTE(odashi):
      // Deleting cl::Buffer does NOT block the process regardless whether the
      // remaining kernel functions are still working or not.
      // We have to manually wait for finishing all kernel functions to prevent
      // memory corruption.
      cmd_queue_.finish();
      // Then, we can delete the buffer safely.
      delete static_cast<cl::Buffer *>(ptr);
  });
}

std::vector<float> OpenCL::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t size = x.shape().size();
  std::vector<float> ret(size);
  float *mapped_ptr = static_cast<float *>(
      cmd_queue_.enqueueMapBuffer(
        ::get_buffer(x), CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0));
  std::memcpy(ret.data(), mapped_ptr, sizeof(float) * size);
  cmd_queue_.enqueueUnmapMemObject(::get_buffer(x), mapped_ptr);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmax_impl(
    const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = std::min(argmax_kernel_group_size_, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::Buffer py = cl::Buffer(
      context_, CL_MEM_WRITE_ONLY, sizeof(std::uint32_t) * r, NULL);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmax_kernel_[m].setArg(0, ::get_buffer(x)); \
      argmax_kernel_[m].setArg(1, s); \
      argmax_kernel_[m].setArg(2, n); \
      argmax_kernel_[m].setArg(3, py); \
      cmd_queue_.enqueueNDRangeKernel( \
          argmax_kernel_[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
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
  cmd_queue_.enqueueReadBuffer(
      py, CL_TRUE, 0, sizeof(std::uint32_t) * r, ret.data());
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmin_impl(
    const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = std::min(argmin_kernel_group_size_, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::Buffer py = cl::Buffer(
      context_, CL_MEM_WRITE_ONLY, sizeof(std::uint32_t) * r, NULL);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmin_kernel_[m].setArg(0, ::get_buffer(x)); \
      argmin_kernel_[m].setArg(1, s); \
      argmin_kernel_[m].setArg(2, n); \
      argmin_kernel_[m].setArg(3, py); \
      cmd_queue_.enqueueNDRangeKernel( \
          argmin_kernel_[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
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
  cmd_queue_.enqueueReadBuffer(
      py, CL_TRUE, 0, sizeof(std::uint32_t) * r, ret.data());
  return ret;
}

void OpenCL::reset_tensor_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  cmd_queue_.enqueueFillBuffer<float>(
      ::get_buffer(x), k, 0, sizeof(float) * size);
}

void OpenCL::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::uint32_t size = x.shape().size();
  float *mapped_ptr = static_cast<float *>(
      cmd_queue_.enqueueMapBuffer(
        ::get_buffer(x), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  std::memcpy(mapped_ptr, values, sizeof(float) * size);
  cmd_queue_.enqueueUnmapMemObject(::get_buffer(x), mapped_ptr);
}

void OpenCL::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case Device::DeviceType::CPU:
      reset_tensor_by_array(static_cast<const float *>((x).data()), y);
      break;
    case Device::DeviceType::OPENCL:
      if(&x.device() == this) {
        const std::uint32_t size = x.shape().size();
        cmd_queue_.enqueueCopyBuffer(
            ::get_buffer(x), ::get_buffer(y), 0, 0, sizeof(float) * size);
      } else {
        reset_tensor_by_vector(x.to_vector(), y);
      }
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void OpenCL::identity_impl(Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, set_identity_kernel_group_size_);
  set_identity_kernel_.setArg(0, size);
  set_identity_kernel_.setArg(1, skip);
  set_identity_kernel_.setArg(2, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      set_identity_kernel_, cl::NullRange,
      cl::NDRange(num_blocks * set_identity_kernel_group_size_),
      cl::NDRange(set_identity_kernel_group_size_));
}

void OpenCL::random_bernoulli_impl(float p, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      cmd_queue_.enqueueMapBuffer(
        ::get_buffer(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  randomizer_.fill_bernoulli(p, size, mapped_ptr);
  cmd_queue_.enqueueUnmapMemObject(::get_buffer(y), mapped_ptr);
}

void OpenCL::random_uniform_impl(float lower, float upper, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      cmd_queue_.enqueueMapBuffer(
        ::get_buffer(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  randomizer_.fill_uniform(lower, upper, size, mapped_ptr);
  cmd_queue_.enqueueUnmapMemObject(::get_buffer(y), mapped_ptr);
}

void OpenCL::random_normal_impl(float mean, float sd, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      cmd_queue_.enqueueMapBuffer(
        ::get_buffer(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  randomizer_.fill_normal(mean, sd, size, mapped_ptr);
  cmd_queue_.enqueueUnmapMemObject(::get_buffer(y), mapped_ptr);
}

void OpenCL::random_log_normal_impl(float mean, float sd, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      cmd_queue_.enqueueMapBuffer(
        ::get_buffer(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  randomizer_.fill_log_normal(mean, sd, size, mapped_ptr);
  cmd_queue_.enqueueUnmapMemObject(::get_buffer(y), mapped_ptr);
}

void OpenCL::pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids,
    std::uint32_t dim, Tensor &y) {
  const std::uint32_t wy = y.shape().lower_volume(dim);
  const std::uint32_t wx = wy * x.shape()[dim];
  const std::uint32_t sx = x.shape().has_batch() * x.shape().volume();
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, pick_fw_kernel_group_size_);
  const std::uint32_t bs = y.shape().batch();
  cl::Buffer ids_buf = ::make_readonly_buffer(context_, ids);
  pick_fw_kernel_.setArg(0, ::get_buffer(x));
  pick_fw_kernel_.setArg(1, ids_buf);
  pick_fw_kernel_.setArg(2, wx);
  pick_fw_kernel_.setArg(3, wy);
  pick_fw_kernel_.setArg(4, sx);
  pick_fw_kernel_.setArg(5, si);
  pick_fw_kernel_.setArg(6, sy);
  pick_fw_kernel_.setArg(7, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      pick_fw_kernel_, cl::NullRange,
      cl::NDRange(g1 * pick_fw_kernel_group_size_, bs),
      cl::NDRange(pick_fw_kernel_group_size_, 1));
}

void OpenCL::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t shift = base * offset;
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, slice_fw_kernel_group_size_);
  slice_fw_kernel_.setArg(0, ::get_buffer(x));
  slice_fw_kernel_.setArg(1, shift);
  slice_fw_kernel_.setArg(2, span);
  slice_fw_kernel_.setArg(3, skip);
  slice_fw_kernel_.setArg(4, size);
  slice_fw_kernel_.setArg(5, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      slice_fw_kernel_, cl::NullRange,
      cl::NDRange(num_blocks * slice_fw_kernel_group_size_),
      cl::NDRange(slice_fw_kernel_group_size_));
}

void OpenCL::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / skip;
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = base * x->shape()[dim];
    const std::uint32_t x_size = span * repeat * x->shape().batch();
    const std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = ::calc_num_blocks(
        y_size, concat_fw_kernel_group_size_);
    concat_fw_kernel_.setArg(0, ::get_buffer(*x));
    concat_fw_kernel_.setArg(1, span);
    concat_fw_kernel_.setArg(2, skip);
    concat_fw_kernel_.setArg(3, x_size);
    concat_fw_kernel_.setArg(4, y_size);
    concat_fw_kernel_.setArg(5, ::get_buffer(y));
    concat_fw_kernel_.setArg(6, offset);
    cmd_queue_.enqueueNDRangeKernel(
        concat_fw_kernel_, cl::NullRange,
        cl::NDRange(num_blocks * concat_fw_kernel_group_size_),
        cl::NDRange(concat_fw_kernel_group_size_), NULL, NULL);
    offset += span;
  }
}

void OpenCL::pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t> &ids,
    std::uint32_t dim, Tensor &gx) {
  const std::uint32_t wy = gy.shape().lower_volume(dim);
  const std::uint32_t wx = wy * gx.shape()[dim];
  const std::uint32_t sx = gx.shape().has_batch() * gx.shape().volume();
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, concat_fw_kernel_group_size_);
  const std::uint32_t bs = gy.shape().batch();
  cl::Buffer ids_buf = ::make_readonly_buffer(context_, ids);
  pick_bw_kernel_.setArg(0, ::get_buffer(gy));
  pick_bw_kernel_.setArg(1, ids_buf);
  pick_bw_kernel_.setArg(2, wx);
  pick_bw_kernel_.setArg(3, wy);
  pick_bw_kernel_.setArg(4, sx);
  pick_bw_kernel_.setArg(5, si);
  pick_bw_kernel_.setArg(6, sy);
  pick_bw_kernel_.setArg(7, ::get_buffer(gx));
  cmd_queue_.enqueueNDRangeKernel(
      pick_bw_kernel_, cl::NullRange,
      cl::NDRange(g1 * concat_fw_kernel_group_size_, bs),
      cl::NDRange(concat_fw_kernel_group_size_, 1));
}

void OpenCL::slice_bw_impl(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  const Shape &sx = gx.shape();
  const Shape &sy = gy.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  const std::uint32_t ox = base * offset;
  const std::uint32_t wx = base * sx[dim];
  const std::uint32_t wy = base * sy[dim];
  const std::uint32_t repeat = sx.volume() / wx;
  const std::uint32_t nx = repeat * sx.batch();
  const std::uint32_t ny = repeat * sy.batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      wy * std::max(nx, ny), slice_bw_kernel_group_size_);
  slice_bw_kernel_.setArg(0, ::get_buffer(gy));
  slice_bw_kernel_.setArg(1, wx);
  slice_bw_kernel_.setArg(2, wy);
  slice_bw_kernel_.setArg(3, nx);
  slice_bw_kernel_.setArg(4, ny);
  slice_bw_kernel_.setArg(5, ::get_buffer(gx));
  slice_bw_kernel_.setArg(6, ox);
  cmd_queue_.enqueueNDRangeKernel(
      slice_bw_kernel_, cl::NullRange,
      cl::NDRange(g1 * slice_bw_kernel_group_size_),
      cl::NDRange(slice_bw_kernel_group_size_));
}

#define OPENCLDEV_FW_X(name) \
void OpenCL::name##_fw_impl(const Tensor &x, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, name##_fw_kernel_group_size_); \
  name##_fw_kernel_.setArg(0, ::get_buffer(x)); \
  name##_fw_kernel_.setArg(1, size); \
  name##_fw_kernel_.setArg(2, ::get_buffer(y)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_fw_kernel_, cl::NullRange, \
      cl::NDRange(num_blocks * name##_fw_kernel_group_size_), \
      cl::NDRange(name##_fw_kernel_group_size_)); \
}

#define OPENCLDEV_BW_X(name) \
void OpenCL::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, name##_bw_kernel_group_size_); \
  name##_bw_kernel_.setArg(0, ::get_buffer(x)); \
  name##_bw_kernel_.setArg(1, ::get_buffer(y)); \
  name##_bw_kernel_.setArg(2, ::get_buffer(gy)); \
  name##_bw_kernel_.setArg(3, size); \
  name##_bw_kernel_.setArg(4, ::get_buffer(gx)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_bw_kernel_, cl::NullRange, \
      cl::NDRange(num_blocks * name##_bw_kernel_group_size_), \
      cl::NDRange(name##_bw_kernel_group_size_)); \
}

#define OPENCLDEV_FW_X_CONST(name) \
void OpenCL::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, name##_fw_kernel_group_size_); \
  name##_fw_kernel_.setArg(0, ::get_buffer(x)); \
  name##_fw_kernel_.setArg(1, k); \
  name##_fw_kernel_.setArg(2, size); \
  name##_fw_kernel_.setArg(3, ::get_buffer(y)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_fw_kernel_, cl::NullRange, \
      cl::NDRange(num_blocks * name##_fw_kernel_group_size_), \
      cl::NDRange(name##_fw_kernel_group_size_)); \
}

#define OPENCLDEV_BW_X_CONST(name) \
void OpenCL::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, name##_bw_kernel_group_size_); \
  name##_bw_kernel_.setArg(0, ::get_buffer(x)); \
  name##_bw_kernel_.setArg(1, ::get_buffer(y)); \
  name##_bw_kernel_.setArg(2, ::get_buffer(gy)); \
  name##_bw_kernel_.setArg(3, k); \
  name##_bw_kernel_.setArg(4, size); \
  name##_bw_kernel_.setArg(5, ::get_buffer(gx)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_bw_kernel_, cl::NullRange, \
      cl::NDRange(num_blocks * name##_bw_kernel_group_size_), \
      cl::NDRange(name##_bw_kernel_group_size_)); \
}

#define OPENCLDEV_FW_X_SCALAR(name) \
void OpenCL::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, name##_fw_kernel_group_size_); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mbx = x.shape().has_batch(); \
  const std::uint32_t mbk = k.shape().has_batch(); \
  name##_fw_kernel_.setArg(0, ::get_buffer(x)); \
  name##_fw_kernel_.setArg(1, ::get_buffer(k)); \
  name##_fw_kernel_.setArg(2, size); \
  name##_fw_kernel_.setArg(3, mbx); \
  name##_fw_kernel_.setArg(4, mbk); \
  name##_fw_kernel_.setArg(5, ::get_buffer(y)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_fw_kernel_, cl::NullRange, \
      cl::NDRange(g1 * name##_fw_kernel_group_size_, g2, 1), \
      cl::NDRange(name##_fw_kernel_group_size_, 1, 1)); \
}

#define OPENCLDEV_FW_AB(name) \
void OpenCL::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, name##_fw_kernel_group_size_); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mba = a.shape().has_batch(); \
  const std::uint32_t mbb = b.shape().has_batch(); \
  name##_fw_kernel_.setArg(0, ::get_buffer(a)); \
  name##_fw_kernel_.setArg(1, ::get_buffer(b)); \
  name##_fw_kernel_.setArg(2, size); \
  name##_fw_kernel_.setArg(3, mba); \
  name##_fw_kernel_.setArg(4, mbb); \
  name##_fw_kernel_.setArg(5, ::get_buffer(y)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_fw_kernel_, cl::NullRange, \
      cl::NDRange(g1 * name##_fw_kernel_group_size_, g2, 1), \
      cl::NDRange(name##_fw_kernel_group_size_, 1, 1)); \
}

#define OPENCLDEV_BW_AB(name) \
void OpenCL::name##_bw_impl( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, name##_bw_kernel_group_size_); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mba = a.shape().has_batch(); \
  const std::uint32_t mbb = b.shape().has_batch(); \
  name##_bw_kernel_.setArg(0, ::get_buffer(a)); \
  name##_bw_kernel_.setArg(1, ::get_buffer(b)); \
  name##_bw_kernel_.setArg(2, ::get_buffer(y)); \
  name##_bw_kernel_.setArg(3, ::get_buffer(gy)); \
  name##_bw_kernel_.setArg(4, size); \
  name##_bw_kernel_.setArg(5, mba); \
  name##_bw_kernel_.setArg(6, mbb); \
  name##_bw_kernel_.setArg(7, ::get_buffer(ga)); \
  name##_bw_kernel_.setArg(8, ::get_buffer(gb)); \
  cmd_queue_.enqueueNDRangeKernel( \
      name##_bw_kernel_, cl::NullRange, \
      cl::NDRange(g1 * name##_bw_kernel_group_size_, g2, 1), \
      cl::NDRange(name##_bw_kernel_group_size_, 1, 1)); \
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
  const std::uint32_t rows = x.shape()[0];
  const std::uint32_t cols = x.shape()[1];
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      rows, transpose_fw_kernel_group_size_x_);
  const std::uint32_t g2 = ::calc_num_blocks(
      cols, transpose_fw_kernel_group_size_y_);
  transpose_fw_kernel_.setArg(0, ::get_buffer(x));
  transpose_fw_kernel_.setArg(1, rows);
  transpose_fw_kernel_.setArg(2, cols);
  transpose_fw_kernel_.setArg(3, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      transpose_fw_kernel_, cl::NullRange,
      cl::NDRange(
        g1 * transpose_fw_kernel_group_size_x_,
        g2 * transpose_fw_kernel_group_size_y_, bs),
      cl::NDRange(
        transpose_fw_kernel_group_size_x_,
        transpose_fw_kernel_group_size_y_, 1));
}

void OpenCL::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  float alpha = 1.;
  float beta = 0.;
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      clblasSgemm(
          clblasColumnMajor, clblasNoTrans, clblasNoTrans,
          di, dk, dj,
          alpha,
          ::get_buffer(a)(), n * a_skip, di,
          ::get_buffer(b)(), n * b_skip, dj,
          beta,
          ::get_buffer(y)(), n * y_skip, di,
          1, &cmd_queue_(), 0, NULL, NULL);
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    clblasSgemm(
        clblasColumnMajor, clblasNoTrans, clblasNoTrans,
        di, dk * b.shape().batch(), dj,
        alpha,
        ::get_buffer(a)(), 0, di,
        ::get_buffer(b)(), 0, dj,
        beta,
        ::get_buffer(y)(), 0, di,
        1, &cmd_queue_(), 0, NULL, NULL);
  }
}

void OpenCL::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const std::uint32_t rows = gx.shape()[0];
  const std::uint32_t cols = gx.shape()[1];
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      rows, transpose_bw_kernel_group_size_x_);
  const std::uint32_t g2 = ::calc_num_blocks(
      cols, transpose_bw_kernel_group_size_y_);
  transpose_bw_kernel_.setArg(0, ::get_buffer(gy));
  transpose_bw_kernel_.setArg(1, rows);
  transpose_bw_kernel_.setArg(2, cols);
  transpose_bw_kernel_.setArg(3, ::get_buffer(gx));
  cmd_queue_.enqueueNDRangeKernel(
      transpose_bw_kernel_, cl::NullRange,
      cl::NDRange(
        g1 * transpose_bw_kernel_group_size_x_,
        g2 * transpose_bw_kernel_group_size_y_, bs),
      cl::NDRange(
        transpose_bw_kernel_group_size_x_,
        transpose_bw_kernel_group_size_y_, 1));
}

void OpenCL::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // ga += gy . b^T
  // gb += a^T . gy
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  float alpha = 1.;
  float beta = 1.;
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      clblasSgemm(
          clblasColumnMajor, clblasNoTrans, clblasTrans,
          di, dj, dk,
          alpha,
          ::get_buffer(gy)(), n * y_skip, di,
          ::get_buffer(b)(), n * b_skip, dj,
          beta,
          ::get_buffer(ga)(), n * a_skip, di,
          1, &cmd_queue_(), 0, NULL, NULL);
      clblasSgemm(
          clblasColumnMajor, clblasTrans, clblasNoTrans,
          dj, dk, di,
          alpha,
          ::get_buffer(a)(), n * a_skip, di,
          ::get_buffer(gy)(), n * y_skip, di,
          beta,
          ::get_buffer(gb)(), n * b_skip, dj,
          1, &cmd_queue_(), 0, NULL, NULL);
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    clblasSgemm(
        clblasColumnMajor, clblasNoTrans, clblasTrans,
        di, dj, dk * b.shape().batch(),
        alpha,
        ::get_buffer(gy)(), 0, di,
        ::get_buffer(b)(), 0, dj,
        beta,
        ::get_buffer(ga)(), 0, di,
        1, &cmd_queue_(), 0, NULL, NULL);
    clblasSgemm(
        clblasColumnMajor, clblasTrans, clblasNoTrans,
        dj, dk * b.shape().batch(), di,
        alpha,
        ::get_buffer(a)(), 0, di,
        ::get_buffer(gy)(), 0, di,
        beta,
        ::get_buffer(gb)(), 0, dj,
        1, &cmd_queue_(), 0, NULL, NULL);
  }
}

void OpenCL::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(sum_fw_kernel_group_size_, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      sum_fw_kernel_[m].setArg(0, ::get_buffer(x)); \
      sum_fw_kernel_[m].setArg(1, s); \
      sum_fw_kernel_[m].setArg(2, n); \
      sum_fw_kernel_[m].setArg(3, ::get_buffer(y)); \
      cmd_queue_.enqueueNDRangeKernel( \
          sum_fw_kernel_[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
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
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(logsumexp_fw_kernel_group_size_, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      logsumexp_fw_kernel_[m].setArg(0, ::get_buffer(x)); \
      logsumexp_fw_kernel_[m].setArg(1, s); \
      logsumexp_fw_kernel_[m].setArg(2, n); \
      logsumexp_fw_kernel_[m].setArg(3, ::get_buffer(y)); \
      cmd_queue_.enqueueNDRangeKernel( \
          logsumexp_fw_kernel_[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
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

void OpenCL::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  const std::uint32_t total = y.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      total, broadcast_fw_kernel_group_size_);
  broadcast_fw_kernel_.setArg(0, ::get_buffer(x));
  broadcast_fw_kernel_.setArg(1, skip1);
  broadcast_fw_kernel_.setArg(2, skip2);
  broadcast_fw_kernel_.setArg(3, total);
  broadcast_fw_kernel_.setArg(4, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      broadcast_fw_kernel_, cl::NullRange,
      cl::NDRange(g1 * broadcast_fw_kernel_group_size_),
      cl::NDRange(broadcast_fw_kernel_group_size_));
}

void OpenCL::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t batch = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, batch_sum_fw_kernel_group_size_);
  batch_sum_fw_kernel_.setArg(0, ::get_buffer(x));
  batch_sum_fw_kernel_.setArg(1, size);
  batch_sum_fw_kernel_.setArg(2, batch);
  batch_sum_fw_kernel_.setArg(3, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      batch_sum_fw_kernel_, cl::NullRange,
      cl::NDRange(g1 * batch_sum_fw_kernel_group_size_),
      cl::NDRange(batch_sum_fw_kernel_group_size_));
}

void OpenCL::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, inplace_multiply_const_kernel_group_size_);
  inplace_multiply_const_kernel_.setArg(0, k);
  inplace_multiply_const_kernel_.setArg(1, size);
  inplace_multiply_const_kernel_.setArg(2, ::get_buffer(x));
  cmd_queue_.enqueueNDRangeKernel(
      inplace_multiply_const_kernel_, cl::NullRange,
      cl::NDRange(g1 * inplace_multiply_const_kernel_group_size_),
      cl::NDRange(inplace_multiply_const_kernel_group_size_));
}

void OpenCL::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t mbx = x.shape().has_batch();
  const std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, inplace_add_kernel_group_size_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  inplace_add_kernel_.setArg(0, ::get_buffer(x));
  inplace_add_kernel_.setArg(1, size);
  inplace_add_kernel_.setArg(2, mbx);
  inplace_add_kernel_.setArg(3, mby);
  inplace_add_kernel_.setArg(4, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      inplace_add_kernel_, cl::NullRange,
      cl::NDRange(g1 * inplace_add_kernel_group_size_, bs, 1),
      cl::NDRange(inplace_add_kernel_group_size_, 1, 1));
}

void OpenCL::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t mbx = x.shape().has_batch();
  const std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, inplace_subtract_kernel_group_size_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  inplace_subtract_kernel_.setArg(0, ::get_buffer(x));
  inplace_subtract_kernel_.setArg(1, size);
  inplace_subtract_kernel_.setArg(2, mbx);
  inplace_subtract_kernel_.setArg(3, mby);
  inplace_subtract_kernel_.setArg(4, ::get_buffer(y));
  cmd_queue_.enqueueNDRangeKernel(
      inplace_subtract_kernel_, cl::NullRange,
      cl::NDRange(g1 * inplace_subtract_kernel_group_size_, bs, 1),
      cl::NDRange(inplace_subtract_kernel_group_size_, 1, 1));
}

}  // namespace devices
}  // namespace primitiv
