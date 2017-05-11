#include <config.h>

#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>
#include <primitiv/tensor_ops.h>

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

class TensorOpsTest : public testing::Test {
protected:
  virtual void SetUp() override {
    device.reset(new CPUDevice());
  }

  std::shared_ptr<Device> device;
};

TEST_F(TensorOpsTest, CheckAddConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
  const Tensor x(Shape({2, 2}, 2), device, x_data);
  const Tensor y1 = k + x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = x + k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckAdd) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
  const Tensor a(Shape({2, 2}, 2), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a + b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b + a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckAddBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
  const Tensor a(Shape({2, 2}), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a + b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b + a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckInvalidAdd) {
  {
    const Tensor a(Shape({2, 2}), device);
    const Tensor b(Shape({3, 3}), device);
    EXPECT_THROW(a + b, std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({2, 2}, 3), device);
    EXPECT_THROW(a + b, std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({3, 3}, 2), device);
    EXPECT_THROW(a + b, std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({3, 3}, 3), device);
    EXPECT_THROW(a + b, std::runtime_error);
  }
}

TEST_F(TensorOpsTest, CheckSubtractConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
  const Tensor x(Shape({2, 2}, 2), device, x_data);
  const Tensor y1 = k - x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = x - k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckSubtract) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y1_data {1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006};
  const vector<float> y2_data {-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006};
  const Tensor a(Shape({2, 2}, 2), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a - b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b - a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckSubtractBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y1_data {0, 1, 2, 3, -4, -3, -2, -1};
  const vector<float> y2_data {0, -1, -2, -3, 4, 3, 2, 1};
  const Tensor a(Shape({2, 2}), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a - b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b - a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckInvalidSubtract) {
  {
    const Tensor a(Shape({2, 2}), device);
    const Tensor b(Shape({3, 3}), device);
    EXPECT_THROW(a - b, std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({2, 2}, 3), device);
    EXPECT_THROW(a - b, std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({3, 3}, 2), device);
    EXPECT_THROW(a - b, std::runtime_error);
  }
  {
    const Tensor a(Shape({2, 2}, 2), device);
    const Tensor b(Shape({3, 3}, 3), device);
    EXPECT_THROW(a - b, std::runtime_error);
  }
}

}  // namespace primitiv
