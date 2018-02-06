#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::random_bernoulli_impl(float p, Tensor &y) {
  randomizer_.fill_bernoulli(p, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
