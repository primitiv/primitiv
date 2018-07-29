#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(
    softplus, (x > 0.).select(
      x + (1. + (-x).exp()).log(),
      (1. + x.exp()).log()));
EIGEN_DEV_BW_X(softplus, gy * (.5 + .5 * (.5 * x).tanh()));

}  // namespace devices
}  // namespace primitiv
