#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/constant_initializer.h>
#include <primitiv/cpu_device.h>
#include <primitiv/shape.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {

class ConstantInitializerTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(ConstantInitializerTest, CheckGenerate) {
  Shape shape {3, 3, 3};
  for (const float k : {1, 10, 100, 1000, 10000}) {
    const vector<float> expected(shape.size(), k);
    const ConstantInitializer init(k);
    EXPECT_EQ(expected, init.generate(shape, &dev).to_vector());
  }
}

}  // namespace primitiv
