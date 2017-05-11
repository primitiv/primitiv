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

TEST_F(TensorOpsTest, CheckMultiplyConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y_data {10000, -1000, 100, -10, 1, -0.1, 0.01, -0.001};
  const Tensor x(Shape({2, 2}, 2), device, x_data);
  const Tensor y1 = k * x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = x * k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckMultiply) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {0, 1, 2, 3, -4, -5, -6, -7};
  const vector<float> y_data {0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007};
  const Tensor a(Shape({2, 2}, 2), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a * b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b * a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckMultiplyBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1 ,1, 1, 1, 0, 1, 2, 3};
  const vector<float> y_data {0, 1, 2, 3, 0, 1, 4, 9};
  const Tensor a(Shape({2, 2}), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a * b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b * a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckDivideConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y1_data {0.01, -0.1, 1, -10, 100, -1000, 10000, -100000};
  const vector<float> y2_data {
    100, -10, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001,
  };
  const Tensor x(Shape({2, 2}, 2), device, x_data);
  const Tensor y1 = k / x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = x / k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckDivide) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {1, 2, 3, 4, -5, -6, -7, -8};
  const vector<float> y1_data {
    1000, -50, 3.33333333, -0.25, -0.02, 0.00166666667, -1.42857143e-4, 1.25e-5,
  };
  const vector<float> y2_data {0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000};
  const Tensor a(Shape({2, 2}, 2), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a / b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b / a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckDivideBatchBroadcast) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {1, 1, 1, 1, 1, 2, 3, 4};
  const vector<float> y1_data {1, 2, 3, 4, 1, 1, 1, 1};
  const vector<float> y2_data {1, 0.5, 0.333333333, 0.25, 1, 1, 1, 1};
  const Tensor a(Shape({2, 2}), device, a_data);
  const Tensor b(Shape({2, 2}, 2), device, b_data);
  const Tensor y1 = a / b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b / a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckInvalidArithmeticOps) {
  const vector<Shape> sa {
    Shape({2, 2}, 2), Shape({2, 2}, 2), Shape({2, 2}, 2),
  };
  const vector<Shape> sb {
    Shape({2, 2}, 3), Shape({3, 3}, 2), Shape({3, 3}, 3),
  };
  for (unsigned i = 0; i < sa.size(); ++i) {
    const Tensor a(sa[i], device, vector<float>(sa[i].size()));
    const Tensor b(sb[i], device, vector<float>(sb[i].size()));
    EXPECT_THROW(a + b, std::runtime_error);
    EXPECT_THROW(a - b, std::runtime_error);
    EXPECT_THROW(a * b, std::runtime_error);
    EXPECT_THROW(a / b, std::runtime_error);
  }
}

}  // namespace primitiv
