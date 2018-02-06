#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/common_naive.h>

namespace primitiv {
namespace devices {

void Naive::random_uniform_impl(float lower, float upper, Tensor &y) {
  randomizer_.fill_uniform(lower, upper, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
