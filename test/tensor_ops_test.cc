#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>
#include <primitiv/tensor_ops.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {
namespace tensor_ops {

class TensorOpsTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TensorOpsTest, CheckAddConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
  const Tensor x(Shape({2, 2}, 2), &dev, x_data);
  const Tensor y1 = k + x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = x + k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckAdd) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
  const Tensor a(Shape({2, 2}, 2), &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a + b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b + a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckAddBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
  const Tensor a(Shape {2, 2}, &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a + b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b + a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckSubtractConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
  const Tensor x(Shape({2, 2}, 2), &dev, x_data);
  const Tensor y1 = k - x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = x - k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckSubtract) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y1_data {1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006};
  const vector<float> y2_data {-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006};
  const Tensor a(Shape({2, 2}, 2), &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a - b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b - a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckSubtractBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y1_data {0, 1, 2, 3, -4, -3, -2, -1};
  const vector<float> y2_data {0, -1, -2, -3, 4, 3, 2, 1};
  const Tensor a(Shape {2, 2}, &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a - b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b - a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckMultiplyConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y_data {10000, -1000, 100, -10, 1, -0.1, 0.01, -0.001};
  const Tensor x(Shape({2, 2}, 2), &dev, x_data);
  const Tensor y1 = k * x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = x * k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckMultiply) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {0, 1, 2, 3, -4, -5, -6, -7};
  const vector<float> y_data {0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007};
  const Tensor a(Shape({2, 2}, 2), &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a * b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b * a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckMultiplyBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 0, 1, 2, 3};
  const vector<float> y_data {0, 1, 2, 3, 0, 1, 4, 9};
  const Tensor a(Shape {2, 2}, &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a * b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
  const Tensor y2 = b * a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckDivideConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y1_data {0.01, -0.1, 1, -10, 100, -1000, 10000, -100000};
  const vector<float> y2_data {
    100, -10, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001,
  };
  const Tensor x(Shape({2, 2}, 2), &dev, x_data);
  const Tensor y1 = k / x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = x / k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckDivide) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {1, 2, 3, 4, -5, -6, -7, -8};
  const vector<float> y1_data {
    1000, -50, 3.33333333, -0.25, -0.02, 0.00166666667, -1.42857143e-4, 1.25e-5,
  };
  const vector<float> y2_data {0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000};
  const Tensor a(Shape({2, 2}, 2), &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a / b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b / a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckDivideBatchBroadcast) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {1, 1, 1, 1, 1, 2, 3, 4};
  const vector<float> y1_data {1, 2, 3, 4, 1, 1, 1, 1};
  const vector<float> y2_data {1, 0.5, 0.333333333, 0.25, 1, 1, 1, 1};
  const Tensor a(Shape {2, 2}, &dev, a_data);
  const Tensor b(Shape({2, 2}, 2), &dev, b_data);
  const Tensor y1 = a / b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y1_data, y1.to_vector()));
  const Tensor y2 = b / a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(test_utils::vector_match(y2_data, y2.to_vector()));
}

TEST_F(TensorOpsTest, CheckInvalidArithmeticOps) {
  const vector<Shape> sa {
    Shape({2, 2}, 2), Shape({2, 2}, 2), Shape({2, 2}, 2),
  };
  const vector<Shape> sb {
    Shape({2, 2}, 3), Shape({3, 3}, 2), Shape({3, 3}, 3),
  };
  for (unsigned i = 0; i < sa.size(); ++i) {
    const Tensor a(sa[i], &dev, vector<float>(sa[i].size()));
    const Tensor b(sb[i], &dev, vector<float>(sb[i].size()));
    EXPECT_THROW(a + b, std::runtime_error);
    EXPECT_THROW(a - b, std::runtime_error);
    EXPECT_THROW(a * b, std::runtime_error);
    EXPECT_THROW(a / b, std::runtime_error);
  }
}

TEST_F(TensorOpsTest, CheckTranspose) {
  {
    const vector<float> x_data {42};
    const vector<float> y_data {42};
    const Tensor x(Shape(), &dev, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape(), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Tensor x(Shape {12}, &dev, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({1, 12}), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Tensor x(Shape({1, 3}, 4), &dev, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3}, 4), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};
    const Tensor x(Shape({2, 2}, 3), &dev, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
    const Tensor x(Shape({2, 3}, 2), &dev, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3, 2}, 2), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidTranspose) {
  const Tensor x(Shape {2, 3, 4}, &dev);
  EXPECT_THROW(transpose(x), std::runtime_error);
}

TEST_F(TensorOpsTest, CheckDot) {
  {
    // A^2
    const vector<float> x_data {1, 2, 3, 4, 1, 0, 0, 1, 0, 2, 3, 0};
    const vector<float> y_data {7, 10, 15, 22, 1, 0, 0, 1, 6, 0, 0, 6};
    const Tensor x(Shape({2, 2}, 3), &dev, x_data);
    const Tensor y = dot(x, x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
  {
    // A . B
    const vector<float> a_data {
      1, 1000, 1,
      10, 100, 100,
      100, 10, 10000,
      1000, 1, 1000000,
    };
    const vector<float> b_data {
      0, 2, 4, 6,
      1, 3, 5, 7,
      8, 6, 4, 2,
      9, 7, 5, 3,
      2, 3, 5, 7,
      9, 4, 1, 0,
    };
    const vector<float> y_data {
      6420, 246, 6040200,
      7531, 1357, 7050301,
      2468, 8642, 2040608,
      3579, 9753, 3050709,
      7532, 2357, 7050302,
      149, 9410, 10409,
    };
    const Tensor a(Shape {3, 4}, &dev, a_data);
    const Tensor b(Shape {4, 6}, &dev, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({3, 6}), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDotBatchBroadcast) {
  {
    const vector<float> a_data {10, 1000, 1, 100};
    const vector<float> b_data {1, 2, 3, 4, 5, 6, 7, 8};
    const vector<float> y_data {12, 1200, 34, 3400, 56, 5600, 78, 7800};
    const Tensor a(Shape {2, 2}, &dev, a_data);
    const Tensor b(Shape({2, 2}, 2), &dev, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
  {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8};
    const vector<float> b_data {10, 1, 1000, 100};
    const vector<float> y_data {13, 24, 1300, 2400, 57, 68, 5700, 6800};
    const Tensor a(Shape({2, 2}, 2), &dev, a_data);
    const Tensor b(Shape {2, 2}, &dev, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(test_utils::vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidDot) {
  {
    // Not a scalar multiplication.
    const Tensor a(Shape {2, 3}, &dev);
    const Tensor b(Shape(), &dev);
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    // Not a scalar multiplication.
    const Tensor a(Shape(), &dev);
    const Tensor b(Shape {2, 3}, &dev);
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    const Tensor a(Shape {2, 3, 4}, &dev);
    const Tensor b(Shape {4}, &dev);
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    const Tensor a(Shape {1, 2}, &dev);
    const Tensor b(Shape {2, 3, 4}, &dev);
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    const Tensor a(Shape {2, 3}, &dev);
    const Tensor b(Shape {2, 3}, &dev);
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
}

TEST_F(TensorOpsTest, CheckTanh) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8
  };
  const vector<float> y_data {
    0, .46211716, .76159416, .96402758, .99932930, .99999977,
    0, -.46211716, -.76159416, -.96402758, -.99932930, -.99999977,
  };
  const Tensor x(Shape({2, 3}, 2), &dev, x_data);
  const Tensor y1 = tanh(x);
  EXPECT_EQ(Shape({2, 3}, 2), y1.shape());
  EXPECT_TRUE(test_utils::vector_match(y_data, y1.to_vector()));
}

}  // namespace tensor_ops
}  // namespace primitiv
