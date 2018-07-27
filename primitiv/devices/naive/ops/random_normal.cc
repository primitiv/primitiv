#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::random_normal_impl(float mean, float sd, Tensor &y) {
  randomizer_.fill_normal(mean, sd, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
