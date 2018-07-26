#include <primitiv/config.h>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::random_normal_impl(float mean, float sd, Tensor &y) {
  randomizer_.fill_normal(mean, sd, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
