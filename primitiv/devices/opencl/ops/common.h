#ifndef PRIMITIV_DEVICES_OPENCL_OPS_COMMON_H_
#define PRIMITIV_DEVICES_OPENCL_OPS_COMMON_H_

#define CDATA(x) (*static_cast<const cl::Buffer *>(get_handle(x)))
#define MDATA(x) (*static_cast<cl::Buffer *>(get_mutable_handle(x)))

#define OPENCLDEV_FW_X(name) \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_fw_kernel.group_size()[0]); \
  state_->name##_fw_kernel.kernel().setArg(0, CDATA(x)); \
  state_->name##_fw_kernel.kernel().setArg(1, size); \
  state_->name##_fw_kernel.kernel().setArg(2, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_fw_kernel.group_size()[0]), \
      cl::NDRange(state_->name##_fw_kernel.group_size()[0]));

#define OPENCLDEV_BW_X(name) \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_bw_kernel.group_size()[0]); \
  state_->name##_bw_kernel.kernel().setArg(0, CDATA(x)); \
  state_->name##_bw_kernel.kernel().setArg(1, CDATA(y)); \
  state_->name##_bw_kernel.kernel().setArg(2, CDATA(gy)); \
  state_->name##_bw_kernel.kernel().setArg(3, size); \
  state_->name##_bw_kernel.kernel().setArg(4, MDATA(gx)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_bw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_bw_kernel.group_size()[0]), \
      cl::NDRange(state_->name##_bw_kernel.group_size()[0]));

#define OPENCLDEV_FW_X_CONST(name) \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_fw_kernel.group_size()[0]); \
  state_->name##_fw_kernel.kernel().setArg(0, CDATA(x)); \
  state_->name##_fw_kernel.kernel().setArg(1, k); \
  state_->name##_fw_kernel.kernel().setArg(2, size); \
  state_->name##_fw_kernel.kernel().setArg(3, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_fw_kernel.group_size()[0]), \
      cl::NDRange(state_->name##_fw_kernel.group_size()[0]));

#define OPENCLDEV_BW_X_CONST(name) \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_bw_kernel.group_size()[0]); \
  state_->name##_bw_kernel.kernel().setArg(0, CDATA(x)); \
  state_->name##_bw_kernel.kernel().setArg(1, CDATA(y)); \
  state_->name##_bw_kernel.kernel().setArg(2, CDATA(gy)); \
  state_->name##_bw_kernel.kernel().setArg(3, k); \
  state_->name##_bw_kernel.kernel().setArg(4, size); \
  state_->name##_bw_kernel.kernel().setArg(5, MDATA(gx)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_bw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_bw_kernel.group_size()[0]), \
      cl::NDRange(state_->name##_bw_kernel.group_size()[0]));

#define OPENCLDEV_FW_X_SCALAR(name) \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, state_->name##_fw_kernel.group_size()[0]); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mbx = x.shape().has_batch(); \
  const std::uint32_t mbk = k.shape().has_batch(); \
  state_->name##_fw_kernel.kernel().setArg(0, CDATA(x)); \
  state_->name##_fw_kernel.kernel().setArg(1, CDATA(k)); \
  state_->name##_fw_kernel.kernel().setArg(2, size); \
  state_->name##_fw_kernel.kernel().setArg(3, mbx); \
  state_->name##_fw_kernel.kernel().setArg(4, mbk); \
  state_->name##_fw_kernel.kernel().setArg(5, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(g1 * state_->name##_fw_kernel.group_size()[0], g2, 1), \
      cl::NDRange(state_->name##_fw_kernel.group_size()[0], 1, 1));

#define OPENCLDEV_FW_AB(name) \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, state_->name##_fw_kernel.group_size()[0]); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mba = a.shape().has_batch(); \
  const std::uint32_t mbb = b.shape().has_batch(); \
  state_->name##_fw_kernel.kernel().setArg(0, CDATA(a)); \
  state_->name##_fw_kernel.kernel().setArg(1, CDATA(b)); \
  state_->name##_fw_kernel.kernel().setArg(2, size); \
  state_->name##_fw_kernel.kernel().setArg(3, mba); \
  state_->name##_fw_kernel.kernel().setArg(4, mbb); \
  state_->name##_fw_kernel.kernel().setArg(5, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(g1 * state_->name##_fw_kernel.group_size()[0], g2, 1), \
      cl::NDRange(state_->name##_fw_kernel.group_size()[0], 1, 1));

#define OPENCLDEV_BW_AB(name) \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, state_->name##_bw_kernel.group_size()[0]); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mba = a.shape().has_batch(); \
  const std::uint32_t mbb = b.shape().has_batch(); \
  state_->name##_bw_kernel.kernel().setArg(0, CDATA(a)); \
  state_->name##_bw_kernel.kernel().setArg(1, CDATA(b)); \
  state_->name##_bw_kernel.kernel().setArg(2, CDATA(y)); \
  state_->name##_bw_kernel.kernel().setArg(3, CDATA(gy)); \
  state_->name##_bw_kernel.kernel().setArg(4, size); \
  state_->name##_bw_kernel.kernel().setArg(5, mba); \
  state_->name##_bw_kernel.kernel().setArg(6, mbb); \
  state_->name##_bw_kernel.kernel().setArg(7, MDATA(ga)); \
  state_->name##_bw_kernel.kernel().setArg(8, MDATA(gb)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_bw_kernel.kernel(), cl::NullRange, \
      cl::NDRange(g1 * state_->name##_bw_kernel.group_size()[0], g2, 1), \
      cl::NDRange(state_->name##_bw_kernel.group_size()[0], 1, 1));

namespace {

/**
 * Copies a device buffer to a host array.
 * @param queue cl::CommandQueue object to perform operations.
 * @param buffer cl::Buffer object to be updated.
 * @param data Array of the data.
 * @param size Number of objects in `data`.
 */
template<typename T>
void read_buffer(
    cl::CommandQueue &queue, const cl::Buffer &buffer,
    T data[], std::size_t size) {
  queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * size, data);
}

/**
 * Copies a host array to a device buffer.
 * @param queue cl::CommandQueue object to perform operations.
 * @param buffer cl::Buffer object to be updated.
 * @param data Array of the data.
 * @param size Number of objects in `data`.
 */
template<typename T>
void write_buffer(
    cl::CommandQueue &queue, cl::Buffer &buffer,
    const T data[], std::size_t size) {
  // NOTE(odashi):
  // Some devices could not directly write their buffers.
  // (I observed this issue using Intel GPUs.)
  // For now, we disabled below code,
  //
  //queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * size, data);
  //
  // and enables copying through memory mapping.
  T *mapped_ptr = static_cast<T *>(
      queue.enqueueMapBuffer(
        buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(T) * size, 0));
  std::memcpy(mapped_ptr, data, sizeof(T) * size);
  queue.enqueueUnmapMemObject(buffer, mapped_ptr);
}

/**
 * Obtains the number of blocks in one parallel operation.
 * @param size Total number of threads.
 * @param num_threads Number of threads in one block.
 */
inline std::uint32_t calc_num_blocks(std::uint32_t size, std::uint32_t num_threads) {
  return (size + num_threads - 1) / num_threads;
}

/**
 * Obtains mutable cl::Buffer from shared_ptr<void>.
 * @param ptr Target shared_ptr object.
 * @return cl::Buffer object which the shared_ptr holds.
 */
inline cl::Buffer &get_buffer(std::shared_ptr<void> &ptr) {
  return *static_cast<cl::Buffer *>(ptr.get());
}

}  // namespace

#endif  // PRIMITIV_DEVICES_OPENCL_OPS_COMMON_H_
