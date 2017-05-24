#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>
#include <primitiv/tensor_ops.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {
namespace tensor_ops {

class TensorOpsTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TensorOpsTest, CheckDuplicate) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const Tensor x = dev.new_tensor(Shape({2, 2}, 2), x_data);
  const Tensor y = +x;
  EXPECT_EQ(Shape({2, 2}, 2), y.shape());
  EXPECT_TRUE(vector_match(x_data, y.get_values()));
}

TEST_F(TensorOpsTest, CheckNegate) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> y_data {
    -1000, -100, -10, -1, -0.1, -0.01, -0.001, -0.0001,
  };
  const Tensor x = dev.new_tensor(Shape({2, 2}, 2), x_data);
  const Tensor y = -x;
  EXPECT_EQ(Shape({2, 2}, 2), y.shape());
  EXPECT_TRUE(vector_match(y_data, y.get_values()));
}

TEST_F(TensorOpsTest, CheckAddConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
  const Tensor x = dev.new_tensor(Shape({2, 2}, 2), x_data);
  const Tensor y1 = k + x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
  const Tensor y2 = x + k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckAdd) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
  const Tensor a = dev.new_tensor(Shape({2, 2}, 2), a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a + b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
  const Tensor y2 = b + a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckAddBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
  const Tensor a = dev.new_tensor({2, 2}, a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a + b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
  const Tensor y2 = b + a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckSubtractConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
  const Tensor x = dev.new_tensor(Shape({2, 2}, 2), x_data);
  const Tensor y1 = k - x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y1_data, y1.get_values()));
  const Tensor y2 = x - k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y2_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckSubtract) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y1_data {1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006};
  const vector<float> y2_data {-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006};
  const Tensor a = dev.new_tensor(Shape({2, 2}, 2), a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a - b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y1_data, y1.get_values()));
  const Tensor y2 = b - a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y2_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckSubtractBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y1_data {0, 1, 2, 3, -4, -3, -2, -1};
  const vector<float> y2_data {0, -1, -2, -3, 4, 3, 2, 1};
  const Tensor a = dev.new_tensor({2, 2}, a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a - b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y1_data, y1.get_values()));
  const Tensor y2 = b - a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y2_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckMultiplyConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y_data {10000, -1000, 100, -10, 1, -0.1, 0.01, -0.001};
  const Tensor x = dev.new_tensor(Shape({2, 2}, 2), x_data);
  const Tensor y1 = k * x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
  const Tensor y2 = x * k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckMultiply) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {0, 1, 2, 3, -4, -5, -6, -7};
  const vector<float> y_data {0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007};
  const Tensor a = dev.new_tensor(Shape({2, 2}, 2), a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a * b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
  const Tensor y2 = b * a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckMultiplyBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 0, 1, 2, 3};
  const vector<float> y_data {0, 1, 2, 3, 0, 1, 4, 9};
  const Tensor a = dev.new_tensor({2, 2}, a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a * b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
  const Tensor y2 = b * a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckDivideConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y1_data {0.01, -0.1, 1, -10, 100, -1000, 10000, -100000};
  const vector<float> y2_data {
    100, -10, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001,
  };
  const Tensor x = dev.new_tensor(Shape({2, 2}, 2), x_data);
  const Tensor y1 = k / x;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y1_data, y1.get_values()));
  const Tensor y2 = x / k;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y2_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckDivide) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {1, 2, 3, 4, -5, -6, -7, -8};
  const vector<float> y1_data {
    1000, -50, 3.33333333, -0.25, -0.02, 0.00166666667, -1.42857143e-4, 1.25e-5,
  };
  const vector<float> y2_data {0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000};
  const Tensor a = dev.new_tensor(Shape({2, 2}, 2), a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a / b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y1_data, y1.get_values()));
  const Tensor y2 = b / a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y2_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckDivideBatchBroadcast) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {1, 1, 1, 1, 1, 2, 3, 4};
  const vector<float> y1_data {1, 2, 3, 4, 1, 1, 1, 1};
  const vector<float> y2_data {1, 0.5, 0.333333333, 0.25, 1, 1, 1, 1};
  const Tensor a = dev.new_tensor({2, 2}, a_data);
  const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
  const Tensor y1 = a / b;
  EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y1_data, y1.get_values()));
  const Tensor y2 = b / a;
  EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
  EXPECT_TRUE(vector_match(y2_data, y2.get_values()));
}

TEST_F(TensorOpsTest, CheckInvalidArithmeticOps) {
  const vector<Shape> sa {
    Shape({2, 2}, 2), Shape({2, 2}, 2), Shape({2, 2}, 2),
  };
  const vector<Shape> sb {
    Shape({2, 2}, 3), Shape({3, 3}, 2), Shape({3, 3}, 3),
  };
  for (unsigned i = 0; i < sa.size(); ++i) {
    const Tensor a = dev.new_tensor(sa[i], vector<float>(sa[i].size()));
    const Tensor b = dev.new_tensor(sb[i], vector<float>(sb[i].size()));
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
    const Tensor x = dev.new_tensor({}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape(), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Tensor x = dev.new_tensor({12}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({1, 12}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Tensor x = dev.new_tensor(Shape({1, 3}, 4), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3}, 4), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};
    const Tensor x = dev.new_tensor(Shape({2, 2}, 3), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
  {
    const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> y_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
    const Tensor x = dev.new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidTranspose) {
  const Tensor x = dev.new_tensor({2, 3, 4});
  EXPECT_THROW(transpose(x), std::runtime_error);
}

TEST_F(TensorOpsTest, CheckDot) {
  {
    // A^2
    const vector<float> x_data {1, 2, 3, 4, 1, 0, 0, 1, 0, 2, 3, 0};
    const vector<float> y_data {7, 10, 15, 22, 1, 0, 0, 1, 6, 0, 0, 6};
    const Tensor x = dev.new_tensor(Shape({2, 2}, 3), x_data);
    const Tensor y = dot(x, x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
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
    const Tensor a = dev.new_tensor({3, 4}, a_data);
    const Tensor b = dev.new_tensor({4, 6}, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({3, 6}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
}

TEST_F(TensorOpsTest, CheckDotBatchBroadcast) {
  {
    const vector<float> a_data {10, 1000, 1, 100};
    const vector<float> b_data {1, 2, 3, 4, 5, 6, 7, 8};
    const vector<float> y_data {12, 1200, 34, 3400, 56, 5600, 78, 7800};
    const Tensor a = dev.new_tensor({2, 2}, a_data);
    const Tensor b = dev.new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
  {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8};
    const vector<float> b_data {10, 1, 1000, 100};
    const vector<float> y_data {13, 24, 1300, 2400, 57, 68, 5700, 6800};
    const Tensor a = dev.new_tensor(Shape({2, 2}, 2), a_data);
    const Tensor b = dev.new_tensor({2, 2}, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.get_values()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidDot) {
  {
    // Not a scalar multiplication.
    const Tensor a = dev.new_tensor({2, 3});
    const Tensor b = dev.new_tensor({});
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    // Not a scalar multiplication.
    const Tensor a = dev.new_tensor({});
    const Tensor b = dev.new_tensor({2, 3});
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    const Tensor a = dev.new_tensor({2, 3, 4});
    const Tensor b = dev.new_tensor({4});
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    const Tensor a = dev.new_tensor({1, 2});
    const Tensor b = dev.new_tensor({2, 3, 4});
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
  {
    const Tensor a = dev.new_tensor({2, 3});
    const Tensor b = dev.new_tensor({2, 3});
    EXPECT_THROW(dot(a, b), std::runtime_error);
  }
}

TEST_F(TensorOpsTest, CheckExp) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8
  };
  const vector<float> y_data {
    1, 1.6487213, 2.7182818, 7.3890561, 54.598150, 2980.9580,
    1, .60653066, .36787944, .13533528, .018315639, .00033546263,
  };
  const Tensor x = dev.new_tensor(Shape({2, 3}, 2), x_data);
  const Tensor y1 = exp(x);
  EXPECT_EQ(Shape({2, 3}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
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
  const Tensor x = dev.new_tensor(Shape({2, 3}, 2), x_data);
  const Tensor y1 = tanh(x);
  EXPECT_EQ(Shape({2, 3}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
}

TEST_F(TensorOpsTest, CheckSigmoid) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8
  };
  const vector<float> y_data {
    .5, .62245933, .73105858, .88079708, .98201379, .99966465,
    .5, .37754067, .26894142, .11920292, .017986210, .00033535013,
  };
  const Tensor x = dev.new_tensor(Shape({2, 3}, 2), x_data);
  const Tensor y1 = sigmoid(x);
  EXPECT_EQ(Shape({2, 3}, 2), y1.shape());
  EXPECT_TRUE(vector_match(y_data, y1.get_values()));
}

}  // namespace tensor_ops
}  // namespace primitiv
