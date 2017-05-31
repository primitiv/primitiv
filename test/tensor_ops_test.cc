#include <config.h>

#include <algorithm>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>
#include <primitiv/tensor_ops.h>
#include <test_utils.h>

#ifdef USE_CUDA
#include <primitiv/cuda_device.h>
#endif  // USE_CUDA

using std::vector;
using test_utils::vector_match;

namespace primitiv {
namespace tensor_ops {

class TensorOpsTest : public testing::Test {
protected:
  vector<Device *> devices;

  void SetUp() override {
    devices.emplace_back(new CPUDevice());
#ifdef USE_CUDA
    devices.emplace_back(new CUDADevice(0));
#endif  // USE_CUDA
  }

  void TearDown() override {
    for (Device *dev : devices) {
      delete dev;
    }
  }
};

TEST_F(TensorOpsTest, CheckSlice) {
  vector<float> x_data(3 * 3 * 2 * 2);
  std::iota(x_data.begin(), x_data.end(), 0);
  struct TestCase {
    unsigned dim, lower, upper;
    Shape shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    // leftmost
    {0, 0, 1, Shape({1, 3, 2}, 2),
      {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33}},
    {1, 0, 1, Shape({3, 1, 2}, 2),
      {0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29}},
    {2, 0, 1, Shape({3, 3, 1}, 2),
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26}},
    // middle
    {0, 1, 2, Shape({1, 3, 2}, 2),
      {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34}},
    {1, 1, 2, Shape({3, 1, 2}, 2),
      {3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32}},
    {2, 1, 2, Shape({3, 3, 1}, 2),
      {9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35}},
    // rightmost
    {0, 2, 3, Shape({1, 3, 2}, 2),
      {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}},
    {1, 2, 3, Shape({3, 1, 2}, 2),
      {6, 7, 8, 15, 16, 17, 24, 25, 26, 33, 34, 35}},
    // higher dim
    {3, 0, 1, Shape({3, 3, 2}, 2), x_data},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({3, 3, 2}, 2), x_data);
    for (const TestCase &tc : test_cases) {
      const Tensor y = slice(x, tc.dim, tc.lower, tc.upper);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidSlice) {
  struct TestCase { unsigned dim, lower, upper; };
  vector<TestCase> test_cases {
    {0, 0, 0}, {0, 1, 0}, {0, 0, 4}, {0, 3, 4},
    {1, 0, 0}, {1, 1, 0}, {1, 0, 4}, {1, 3, 4},
    {2, 0, 0}, {2, 1, 0}, {2, 0, 2}, {2, 1, 2},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({3, 3}, 3);
    for (const TestCase &tc : test_cases) {
      EXPECT_THROW(slice(x, tc.dim, tc.lower, tc.upper), std::runtime_error);
    }
  }
}

TEST_F(TensorOpsTest, CheckDuplicate) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 2), x_data);
    const Tensor y = +x;
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(x_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckNegate) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> y_data {
    -1000, -100, -10, -1, -0.1, -0.01, -0.001, -0.0001,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 2), x_data);
    const Tensor y = -x;
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckAddConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k + x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = x + k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckAdd) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a + b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = b + a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckAddBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor({2, 2}, a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a + b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = b + a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSubtractConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k - x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = x - k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSubtract) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y1_data {1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006};
  const vector<float> y2_data {-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a - b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = b - a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSubtractBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y1_data {0, 1, 2, 3, -4, -3, -2, -1};
  const vector<float> y2_data {0, -1, -2, -3, 4, 3, 2, 1};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor({2, 2}, a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a - b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = b - a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckMultiplyConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y_data {10000, -1000, 100, -10, 1, -0.1, 0.01, -0.001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k * x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = x * k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckMultiply) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {0, 1, 2, 3, -4, -5, -6, -7};
  const vector<float> y_data {0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a * b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = b * a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckMultiplyBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 0, 1, 2, 3};
  const vector<float> y_data {0, 1, 2, 3, 0, 1, 4, 9};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor({2, 2}, a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a * b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = b * a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDivideConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y1_data {0.01, -0.1, 1, -10, 100, -1000, 10000, -100000};
  const vector<float> y2_data {
    100, -10, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k / x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = x / k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDivide) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {1, 2, 3, 4, -5, -6, -7, -8};
  const vector<float> y1_data {
    1000, -50, 3.33333333, -0.25, -0.02, 0.00166666667, -1.42857143e-4, 1.25e-5,
  };
  const vector<float> y2_data {0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a / b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = b / a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDivideBatchBroadcast) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {1, 1, 1, 1, 1, 2, 3, 4};
  const vector<float> y1_data {1, 2, 3, 4, 1, 1, 1, 1};
  const vector<float> y2_data {1, 0.5, 0.333333333, 0.25, 1, 1, 1, 1};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor({2, 2}, a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y1 = a / b;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = b / a;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidArithmeticOps) {
  const vector<Shape> sa {
    Shape({2, 2}, 2), Shape({2, 2}, 2), Shape({2, 2}, 2),
  };
  const vector<Shape> sb {
    Shape({2, 2}, 3), Shape({3, 3}, 2), Shape({3, 3}, 3),
  };
  for (Device *dev : devices) {
    for (unsigned i = 0; i < sa.size(); ++i) {
      const Tensor a = dev->new_tensor(sa[i], vector<float>(sa[i].size()));
      const Tensor b = dev->new_tensor(sb[i], vector<float>(sb[i].size()));
      EXPECT_THROW(a + b, std::runtime_error);
      EXPECT_THROW(a - b, std::runtime_error);
      EXPECT_THROW(a * b, std::runtime_error);
      EXPECT_THROW(a / b, std::runtime_error);
    }
  }
}

TEST_F(TensorOpsTest, CheckTranspose11) {
  for (Device *dev : devices) {
    const vector<float> x_data {42};
    const vector<float> y_data {42};
    const Tensor x = dev->new_tensor({}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape(), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTransposeN1) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({12}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({1, 12}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTranspose1N) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({1, 3}, 4), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3}, 4), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTransposeNN) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 3), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTransposeMN) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidTranspose) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({2, 3, 4});
    EXPECT_THROW(transpose(x), std::runtime_error);
  }
}

TEST_F(TensorOpsTest, CheckDotAA) {
  const vector<float> x_data {1, 2, 3, 4, 1, 0, 0, 1, 0, 2, 3, 0};
  const vector<float> y_data {7, 10, 15, 22, 1, 0, 0, 1, 6, 0, 0, 6};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2}, 3), x_data);
    const Tensor y = dot(x, x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDotAB) {
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
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor({3, 4}, a_data);
    const Tensor b = dev->new_tensor({4, 6}, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({3, 6}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDotBatchBroadcast1N) {
  const vector<float> a_data {10, 1000, 1, 100};
  const vector<float> b_data {1, 2, 3, 4, 5, 6, 7, 8};
  const vector<float> y_data {12, 1200, 34, 3400, 56, 5600, 78, 7800};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor({2, 2}, a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 2), b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDotBatchBroadcastN1) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8};
  const vector<float> b_data {10, 1, 1000, 100};
  const vector<float> y_data {13, 24, 1300, 2400, 57, 68, 5700, 6800};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor({2, 2}, b_data);
    const Tensor y = dot(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidDot) {
  for (Device *dev : devices) {
    {
      // Not a scalar multiplication.
      const Tensor a = dev->new_tensor({2, 3});
      const Tensor b = dev->new_tensor({});
      EXPECT_THROW(dot(a, b), std::runtime_error);
    }
    {
      // Not a scalar multiplication.
      const Tensor a = dev->new_tensor({});
      const Tensor b = dev->new_tensor({2, 3});
      EXPECT_THROW(dot(a, b), std::runtime_error);
    }
    {
      const Tensor a = dev->new_tensor({2, 3, 4});
      const Tensor b = dev->new_tensor({4});
      EXPECT_THROW(dot(a, b), std::runtime_error);
    }
    {
      const Tensor a = dev->new_tensor({1, 2});
      const Tensor b = dev->new_tensor({2, 3, 4});
      EXPECT_THROW(dot(a, b), std::runtime_error);
    }
    {
      const Tensor a = dev->new_tensor({2, 3});
      const Tensor b = dev->new_tensor({2, 3});
      EXPECT_THROW(dot(a, b), std::runtime_error);
    }
  }
}

TEST_F(TensorOpsTest, CheckExp) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    1, 1.6487213, 2.7182818, 7.3890561, 54.598150, 2980.9580,
    1, .60653066, .36787944, .13533528, .018315639, .00033546263,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = exp(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTanh) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0, .46211716, .76159416, .96402758, .99932930, .99999977,
    0, -.46211716, -.76159416, -.96402758, -.99932930, -.99999977,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = tanh(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSigmoid) {
  const vector<float> x_data {
    0, .5, 1, 2, 3, 4,
    0, -.5, -1, -2, -3, -4,
  };
  const vector<float> y_data {
    .5, .62245933, .73105858, .88079708, .95257413, .98201379,
    .5, .37754067, .26894142, .11920292, .047425873, .017986210,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = sigmoid(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckStep) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = step(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckRelu) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0, .5, 1, 2, 4, 8,
    0, 0, 0, 0, 0, 0,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 2), x_data);
    const Tensor y = relu(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckBatchSum) {
  const vector<float> x_data {
    1, 2, 3, 4, 5, 6, 7, 8,
    -2, -4, -6, -8, -10, -12, -14, -16,
  };
  const vector<float> y_data {
    -1, -2, -3, -4, -5, -6, -7, -8,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 2, 2}, 2), x_data);
    const Tensor y = batch_sum(x);
    EXPECT_EQ(Shape({2, 2, 2}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

}  // namespace tensor_ops
}  // namespace primitiv
