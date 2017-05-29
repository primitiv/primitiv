#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/shape.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {
namespace initializers {

class InitializerImplTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(InitializerImplTest, CheckConstant) {
  Shape shape {3, 3, 3};
  for (const float k : {1, 10, 100, 1000, 10000}) {
    const vector<float> expected(shape.size(), k);
    const Constant init(k);
    Tensor x = dev.new_tensor(shape);
    init.apply(x);
    EXPECT_EQ(expected, x.to_vector());
  }
}

}  // namespace initializers
}  // namespace primitiv
