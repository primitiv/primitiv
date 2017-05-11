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
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(::vector_match(y_data, y.to_vector()));
  }
}

TEST_F(CPUDeviceCalcTest, CheckAdd) {
  {
    const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
    const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
    const Tensor a(Shape({2, 2}, 2), device, a_data);
    const Tensor b(Shape({2, 2}, 2), device, b_data);
    const Tensor y = a.device()->add(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> a_data {0, 1, 2, 3};
    const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
    const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
    const Tensor a(Shape({2, 2}), device, a_data);
    const Tensor b(Shape({2, 2}, 2), device, b_data);
    const Tensor y = a.device()->add(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> a_data {0, 0, 0, 0, 4, 4, 4, 4};
    const vector<float> b_data {0, 1, 2, 3};
    const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
    const Tensor a(Shape({2, 2}, 2), device, a_data);
    const Tensor b(Shape({2, 2}), device, b_data);
    const Tensor y = a.device()->add(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(::vector_match(y_data, y.to_vector()));
  }
}

TEST_F(CPUDeviceCalcTest, CheckInvalidAdd) {
  {
    const Tensor a(Shape({2, 2}), device);
    const Tensor b(Shape({3, 3}), device);
    EXPECT_THROW(a.device()->add(a, b), std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({2, 2}, 3), device);
    EXPECT_THROW(a.device()->add(a, b), std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({3, 3}, 2), device);
    EXPECT_THROW(a.device()->add(a, b), std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({3, 3}, 3), device);
    EXPECT_THROW(a.device()->add(a, b), std::runtime_error);
  }
}

}  // namespace primitiv
