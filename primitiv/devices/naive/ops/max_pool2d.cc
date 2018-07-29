#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::max_pool2d_fw_impl(
    const Tensor &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    Tensor &y) {
  const Shape x_shape = x.shape();
  const Shape y_shape = y.shape();

  const std::uint32_t x_height = x_shape[0];
  const std::uint32_t x_width = x_shape[1];
  const std::uint32_t y_height = y_shape[0];
  const std::uint32_t y_width = y_shape[1];

  const std::size_t x_shift = x_height * x_width;
  const std::size_t y_shift = y_height * y_width;

  const std::uint32_t repeat = x_shape.size() / x_shift;

  const float *px = CDATA(x);
  float *py = MDATA(y);

  for (std::uint32_t r = 0; r < repeat; ++r) {
    for (std::uint32_t y_x = 0; y_x < y_width; ++y_x) {
      for (std::uint32_t y_y = 0; y_y < y_height; ++y_y) {
        float maxval = std::numeric_limits<float>::lowest();

        for (std::uint32_t w_x = 0; w_x < window1; ++w_x) {
          const std::int32_t x_x = -padding1 + y_x * stride1 + w_x;
          if (x_x < 0 || x_x >= static_cast<std::int32_t>(x_width)) continue;

          for (std::uint32_t w_y = 0; w_y < window0; ++w_y) {
            const std::int32_t x_y = -padding0 + y_y * stride0 + w_y;
            if (x_y < 0 || x_y >= static_cast<std::int32_t>(x_height)) continue;

            const float val = px[x_x * x_height + x_y];
            if (val > maxval) maxval = val;
          }
        }

        py[y_x * y_height + y_y] = maxval;
      }
    }

    px += x_shift;
    py += y_shift;
  }
}

void Naive::max_pool2d_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    Tensor &gx) {
  const Shape x_shape = x.shape();
  const Shape y_shape = y.shape();

  const std::uint32_t x_height = x_shape[0];
  const std::uint32_t x_width = x_shape[1];
  const std::uint32_t y_height = y_shape[0];
  const std::uint32_t y_width = y_shape[1];

  const std::size_t x_shift = x_height * x_width;
  const std::size_t y_shift = y_height * y_width;

  const std::uint32_t repeat = x_shape.size() / x_shift;

  const float *px = CDATA(x);
  const float *py = CDATA(y);
  const float *pgy = CDATA(gy);
  float *pgx = MDATA(gx);

  for (std::uint32_t r = 0; r < repeat; ++r) {
    for (std::uint32_t y_x = 0; y_x < y_width; ++y_x) {
      for (std::uint32_t y_y = 0; y_y < y_height; ++y_y) {
        const std::uint32_t y_addr = y_x * y_height + y_y;
        const float maxval = py[y_addr];
        const float grad = pgy[y_addr];
        bool next = true;

        for (std::uint32_t w_x = 0; next && w_x < window1; ++w_x) {
          const std::int32_t x_x = -padding1 + y_x * stride1 + w_x;
          if (x_x < 0 || x_x >= static_cast<std::int32_t>(x_width)) continue;

          for (std::uint32_t w_y = 0; next && w_y < window0; ++w_y) {
            const std::int32_t x_y = -padding0 + y_y * stride0 + w_y;
            if (x_y < 0 || x_y >= static_cast<std::int32_t>(x_height)) continue;

            const std::uint32_t x_addr = x_x * x_height + x_y;
            if (px[x_addr] == maxval) {
              pgx[x_addr] += grad;
              next = false;
            }
          }
        }
      }
    }

    px += x_shift;
    py += y_shift;
    pgy += y_shift;
    pgx += x_shift;
  }
}

}  // namespace devices
}  // namespace primitiv
