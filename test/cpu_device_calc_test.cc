#include <config.h>

#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>

using std::vector;

namespace {

// float <-> int32_t bits converter.
union f2i32 {
  float f;
  int32_t i;
};

// check if two float are near than the ULP-based threshold.
bool float_eq(const float a, const float b) {
  static const int MAX_ULPS = 4;
  int ai = reinterpret_cast<const f2i32 *>(&a)->i;
  if (ai < 0) ai = 0x80000000 - ai;
  int bi = reinterpret_cast<const f2i32 *>(&b)->i;
  if (bi < 0) bi = 0x80000000 - bi;
  const int diff = ai > bi ? ai - bi : bi - ai;
  return (diff <= MAX_ULPS);
}

// helper to check vector equality.
::testing::AssertionResult vector_match(
    const vector<float> &expected,
    const vector<float> &actual) {
  if (expected.size() != actual.size()) {
    return ::testing::AssertionFailure()
      << "expected.size() (" << expected.size()
      << ") != actual.size() (" << actual.size() << ")";
  }
  for (unsigned i = 0; i < expected.size(); ++i) {
    if (!::float_eq(expected[i], actual[i])) {
      return ::testing::AssertionFailure()
        << "expected[" << i << "] (" << expected[i]
        << ") != actual[" << i << "] (" << actual[i] << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

namespace primitiv {

class CPUDeviceCalcTest : public testing::Test {
protected:
  virtual void SetUp() override {
    device.reset(new CPUDevice());
  }

  std::shared_ptr<Device> device;
};

TEST_F(CPUDeviceCalcTest, CheckAddConst) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const float k = 1;
    const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
    const Tensor x(Shape({2, 2}, 2), device, x_data);
    const Tensor y = x.device()->add_const(x, k);
    EXPECT_TRUE(::vector_match(y_data, y.to_vector()));
  }
}

}  // namespace primitiv
