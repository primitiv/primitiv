#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::random_bernoulli_impl(float p, Tensor &y) {
  randomizer_.fill_bernoulli(p, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
