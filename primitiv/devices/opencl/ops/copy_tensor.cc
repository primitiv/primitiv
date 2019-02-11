#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case DeviceType::NAIVE:
      reset_tensor_by_array(static_cast<const float *>(get_handle(x)), y);
      break;
    case DeviceType::OPENCL:
      if(&x.device() == this) {
        const std::uint32_t size = x.shape().size();
        state_->queue.enqueueCopyBuffer(
            CDATA(x), MDATA(y), 0, 0, sizeof(float) * size);
      } else {
        const std::uint32_t size = x.shape().size();
        cl::CommandQueue &queue_x = static_cast<OpenCL &>(x.device()).state_->queue;
        const float *mapped_ptr_x = static_cast<const float *>(
            queue_x.enqueueMapBuffer(
              CDATA(x), CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0));
        float *mapped_ptr_y = static_cast<float *>(
            state_->queue.enqueueMapBuffer(
              MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
        std::memcpy(mapped_ptr_y, mapped_ptr_x, sizeof(float) * size);
        queue_x.enqueueUnmapMemObject(
            CDATA(x), const_cast<float *>(mapped_ptr_x));
        state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr_y);
      }
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

}  // namespace devices
}  // namespace primitiv
