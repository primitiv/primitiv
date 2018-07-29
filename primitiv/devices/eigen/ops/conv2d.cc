#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

// TODO(odashi):
// Implove implementation of conv2d.
// These functions are identical with the Naive implementations and extremely
// slow.

void Eigen::conv2d_fw_impl(
    const Tensor &x, const Tensor &w,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1,
    Tensor &y) {
  const Shape x_shape = x.shape();
  const Shape w_shape = w.shape();
  const Shape y_shape = y.shape();

  const std::uint32_t x_height = x_shape[0];
  const std::uint32_t x_width = x_shape[1];
  const std::uint32_t x_channels = x_shape[2];
  const std::uint32_t w_height = w_shape[0];
  const std::uint32_t w_width = w_shape[1];
  const std::uint32_t y_height = y_shape[0];
  const std::uint32_t y_width = y_shape[1];
  const std::uint32_t y_channels = y_shape[2];

  const std::uint32_t batch_size = y_shape.batch();

  const std::size_t x_shift = x_shape.has_batch() * x_shape.volume();
  const std::size_t w_shift = w_shape.has_batch() * w_shape.volume();
  const std::size_t y_shift = y_shape.volume();

  const float *px = CDATA(x);
  const float *pw = CDATA(w);
  float *py = MDATA(y);

  for (std::uint32_t bn = 0; bn < batch_size; ++bn) {
    for (std::uint32_t y_c = 0; y_c < y_channels; ++y_c) {
      for (std::uint32_t y_x = 0; y_x < y_width; ++y_x) {
        for (std::uint32_t y_y = 0; y_y < y_height; ++y_y) {
          const std::uint32_t y_addr = (y_c * y_width + y_x) * y_height + y_y;
          py[y_addr] = 0;

          for (std::uint32_t x_c = 0; x_c < x_channels; ++x_c) {
            for (
                std::uint32_t w_x = 0, w_x_inv = w_width - 1;
                w_x < w_width; ++w_x, --w_x_inv) {
              for (
                  std::uint32_t w_y = 0, w_y_inv = w_height - 1;
                  w_y < w_height; ++w_y, --w_y_inv) {
                const std::int32_t x_y
                  = -padding0 + y_y * stride0 + w_y * dilation0;
                const std::int32_t x_x
                  = -padding1 + y_x * stride1 + w_x * dilation1;

                if (x_y >= 0 && x_y < static_cast<std::int32_t>(x_height)
                    && x_x >= 0 && x_x < static_cast<std::int32_t>(x_width)) {
                  const std::uint32_t x_addr
                    = (x_c * x_width + x_x) * x_height + x_y;
                  const std::uint32_t w_addr
                    = ((y_c * x_channels + x_c) * w_width + w_x_inv)
                    * w_height + w_y_inv;
                  py[y_addr] += px[x_addr] * pw[w_addr];
                }
              }
            }
          }
        }
      }
    }

    px += x_shift;
    pw += w_shift;
    py += y_shift;
  }
}

void Eigen::conv2d_bw_impl(
    const Tensor &x, const Tensor &w, const Tensor &, const Tensor &gy,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    std::uint32_t dilation0, std::uint32_t dilation1,
    Tensor &gx, Tensor &gw) {
  const Shape x_shape = x.shape();
  const Shape w_shape = w.shape();
  const Shape y_shape = gy.shape();

  const std::uint32_t x_height = x_shape[0];
  const std::uint32_t x_width = x_shape[1];
  const std::uint32_t x_channels = x_shape[2];
  const std::uint32_t w_height = w_shape[0];
  const std::uint32_t w_width = w_shape[1];
  const std::uint32_t y_height = y_shape[0];
  const std::uint32_t y_width = y_shape[1];
  const std::uint32_t y_channels = y_shape[2];

  const std::uint32_t batch_size = y_shape.batch();

  const std::size_t x_shift = x_shape.has_batch() * x_shape.volume();
  const std::size_t w_shift = w_shape.has_batch() * w_shape.volume();
  const std::size_t y_shift = y_shape.volume();

  const float *px = CDATA(x);
  const float *pw = CDATA(w);
  const float *pgy = CDATA(gy);
  float *pgx = MDATA(gx);
  float *pgw = MDATA(gw);

  for (std::uint32_t bn = 0; bn < batch_size; ++bn) {
    for (std::uint32_t y_c = 0; y_c < y_channels; ++y_c) {
      for (std::uint32_t y_x = 0; y_x < y_width; ++y_x) {
        for (std::uint32_t y_y = 0; y_y < y_height; ++y_y) {
          const std::uint32_t y_addr = (y_c * y_width + y_x) * y_height + y_y;

          for (std::uint32_t x_c = 0; x_c < x_channels; ++x_c) {
            for (
                std::uint32_t w_x = 0, w_x_inv = w_width - 1;
                w_x < w_width; ++w_x, --w_x_inv) {
              for (
                  std::uint32_t w_y = 0, w_y_inv = w_height - 1;
                  w_y < w_height; ++w_y, --w_y_inv) {
                const std::int32_t x_y
                  = -padding0 + y_y * stride0 + w_y * dilation0;
                const std::int32_t x_x
                  = -padding1 + y_x * stride1 + w_x * dilation1;

                if (x_y >= 0 && x_y < static_cast<std::int32_t>(x_height)
                    && x_x >= 0 && x_x < static_cast<std::int32_t>(x_width)) {
                  const std::uint32_t x_addr
                    = (x_c * x_width + x_x) * x_height + x_y;
                  const std::uint32_t w_addr
                    = ((y_c * x_channels + x_c) * w_width + w_x_inv)
                    * w_height + w_y_inv;
                  pgx[x_addr] += pgy[y_addr] * pw[w_addr];
                  pgw[w_addr] += pgy[y_addr] * px[x_addr];
                }
              }
            }
          }
        }
      }
    }

    px += x_shift;
    pw += w_shift;
    pgy += y_shift;
    pgx += x_shift;
    pgw += w_shift;
  }
}

}  // namespace devices
}  // namespace primitiv
