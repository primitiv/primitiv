#include <config.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>
#include <primitiv/tensor.h>
#include <primitiv/tensor_ops.h>
#include <test_utils.h>

#ifdef USE_CUDA
#include <primitiv/cuda_device.h>
#endif  // USE_CUDA

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {
namespace tensor_ops {

class TensorOpsTest : public testing::Test {
protected:
  vector<Device *> devices;

  void SetUp() override {
    devices.emplace_back(new CPUDevice());
    devices.emplace_back(new CPUDevice()); // other device on the same hardware
#ifdef USE_CUDA
    devices.emplace_back(new CUDADevice(0));
    devices.emplace_back(new CUDADevice(0)); // other device on the same hardware
    if (CUDADevice::num_devices() > 2) {
      devices.emplace_back(new CUDADevice(1));
    }
#endif  // USE_CUDA
  }

  void TearDown() override {
    for (Device *dev : devices) {
      delete dev;
    }
  }
};

TEST_F(TensorOpsTest, CheckCopy) {
  const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), data);
    for (Device *dev2 : devices) {
      const Tensor y = copy(x, dev2);
      EXPECT_EQ(Shape({2, 2}, 3), y.shape());
      EXPECT_TRUE(y.device() != x.device() || y.data() != x.data());
      EXPECT_TRUE(vector_match(data, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidCopy) {
  for (Device *dev : devices) {
    EXPECT_THROW(copy(Tensor(), dev), Error);
  }
}

TEST_F(TensorOpsTest, CheckPickNN) {
  struct TestCase {
    Shape x_shape;
    unsigned dim;
    vector<unsigned> ids;
    Shape y_shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2, 2}, 3), 0, {0, 0, 0},
      Shape({1, 2, 2}, 3),
      {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}},
    {Shape({2, 2, 2}, 3), 0, {1, 0, 1},
      Shape({1, 2, 2}, 3),
      {1, 3, 5, 7, 8, 10, 12, 14, 17, 19, 21, 23}},
    {Shape({2, 2, 2}, 3), 0, {0},
      Shape({1, 2, 2}, 3),
      {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}},
    {{2, 2, 2}, 0, {0, 1, 0},
      Shape({1, 2, 2}, 3),
      {0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6}},
    {Shape({2, 2, 2}, 3), 1, {0, 0, 0},
      Shape({2, 1, 2}, 3),
      {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21}},
    {Shape({2, 2, 2}, 3), 2, {0, 0, 0},
      Shape({2, 2, 1}, 3),
      {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      std::cerr << "x_shape=" << tc.x_shape.to_string()
        << ", dim=" << tc.dim << ", ids=[";
      for (unsigned i = 0; i < tc.ids.size(); ++i) {
        if (i > 0) std::cerr << ',';
        std::cerr << tc.ids[i];
      }
      std::cerr << ']' << std::endl;
      vector<float> x_data(tc.x_shape.num_total_elements());
      iota(x_data.begin(), x_data.end(), 0);
      const Tensor x = dev->new_tensor_by_vector(tc.x_shape, x_data);
      const Tensor y = pick(x, tc.dim, tc.ids);
      EXPECT_EQ(tc.y_shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidPick) {
  struct TestCase {
    unsigned dim;
    vector<unsigned> ids;
  };
  const vector<TestCase> test_cases {
     {0, {}},
     {0, {2}},
     {0, {0, 1}},
     {0, {0, 1, 2}},
     {1, {2}},
     {2, {2}},
     {3, {1}},
  };
  for (Device *dev : devices) {
    std::cout << dev << std::endl;
    const Tensor x = dev->new_tensor(Shape({2, 2, 2}, 3), 0);
    for (const TestCase &tc : test_cases) {
      std::cout << tc.dim << std::endl;
      EXPECT_THROW(pick(x, tc.dim, tc.ids), Error);
    }
  }
}

TEST_F(TensorOpsTest, CheckSlice) {
  vector<float> x_data(3 * 3 * 2 * 4);
  std::iota(x_data.begin(), x_data.end(), 0);
  struct TestCase {
    unsigned dim, lower, upper;
    Shape shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    // leftmost
    {0, 0, 1, Shape({1, 3, 2}, 4),
      {0, 3, 6, 9, 12, 15,
        18, 21, 24, 27, 30, 33,
        36, 39, 42, 45, 48, 51,
        54, 57, 60, 63, 66, 69}},
    {1, 0, 1, Shape({3, 1, 2}, 4),
      {0, 1, 2, 9, 10, 11,
        18, 19, 20, 27, 28, 29,
        36, 37, 38, 45, 46, 47,
        54, 55, 56, 63, 64, 65}},
    {2, 0, 1, Shape({3, 3, 1}, 4),
      {0, 1, 2, 3, 4, 5, 6, 7, 8,
        18, 19, 20, 21, 22, 23, 24, 25, 26,
        36, 37, 38, 39, 40, 41, 42, 43, 44,
        54, 55, 56, 57, 58, 59, 60, 61, 62}},
    // middle
    {0, 1, 2, Shape({1, 3, 2}, 4),
      {1, 4, 7, 10, 13, 16,
        19, 22, 25, 28, 31, 34,
        37, 40, 43, 46, 49, 52,
        55, 58, 61, 64, 67, 70}},
    {1, 1, 2, Shape({3, 1, 2}, 4),
      {3, 4, 5, 12, 13, 14,
        21, 22, 23, 30, 31, 32,
        39, 40, 41, 48, 49, 50,
        57, 58, 59, 66, 67, 68}},
    {2, 1, 2, Shape({3, 3, 1}, 4),
      {9, 10, 11, 12, 13, 14, 15, 16, 17,
        27, 28, 29, 30, 31, 32, 33, 34, 35,
        45, 46, 47, 48, 49, 50, 51, 52, 53,
        63, 64, 65, 66, 67, 68, 69, 70, 71}},
    // rightmost
    {0, 2, 3, Shape({1, 3, 2}, 4),
      {2, 5, 8, 11, 14, 17,
        20, 23, 26, 29, 32, 35,
        38, 41, 44, 47, 50, 53,
        56, 59, 62, 65, 68, 71}},
    {1, 2, 3, Shape({3, 1, 2}, 4),
      {6, 7, 8, 15, 16, 17,
        24, 25, 26, 33, 34, 35,
        42, 43, 44, 51, 52, 53,
        60, 61, 62, 69, 70, 71}},
    // higher dim
    {3, 0, 1, Shape({3, 3, 2}, 4), x_data},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({3, 3, 2}, 4), x_data);
    for (const TestCase &tc : test_cases) {
      std::cerr << "dim=" << tc.dim << ", lower=" << tc.lower
        << ", upper=" << tc.upper << std::endl;
      const Tensor y = slice(x, tc.dim, tc.lower, tc.upper);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidSlice) {
  struct TestCase { unsigned dim, lower, upper; };
  const vector<TestCase> test_cases {
    {0, 0, 0}, {0, 1, 0}, {0, 0, 4}, {0, 3, 4},
    {1, 0, 0}, {1, 1, 0}, {1, 0, 4}, {1, 3, 4},
    {2, 0, 0}, {2, 1, 0}, {2, 0, 2}, {2, 1, 2},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({3, 3}, 3);
    for (const TestCase &tc : test_cases) {
      EXPECT_THROW(slice(x, tc.dim, tc.lower, tc.upper), Error);
    }
  }
}

TEST_F(TensorOpsTest, CheckConcatN_3x3) {
  const vector<float> y_data {
    1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
  };
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({1, 3}, {1, 1, 1});
    const Tensor b = dev->new_tensor_by_vector({2, 3}, {2, 3, 2, 3, 2, 3});
    const Tensor c = dev->new_tensor_by_vector({3, 3}, {4, 5, 6, 4, 5, 6, 4, 5, 6});
    const Tensor y = concat({&a, &b, &c}, 0);
    EXPECT_EQ(Shape({6, 3}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckConcat5x4) {
  const vector<Shape> shapes {
    Shape {20},
    Shape {5, 4},
    Shape {5, 1, 4},
  };
  const vector<float> y_data {
    1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
  };
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({5}, {1, 1, 1, 1, 1});
    const Tensor b = dev->new_tensor_by_vector({5}, {2, 2, 2, 2, 2});
    const Tensor c = dev->new_tensor_by_vector({5}, {3, 3, 3, 3, 3});
    const Tensor d = dev->new_tensor_by_vector({5}, {4, 4, 4, 4, 4});
    for (const unsigned i : {0, 1, 2}) {
      const Tensor y = concat({&a, &b, &c, &d}, i);
      EXPECT_EQ(shapes[i], y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckConcat2_2_2x2) {
  const vector<float> a_data {
    1, 2, 3, 4, 5, 6, 7, 8,
    11, 22, 33, 44, 55, 66, 77, 88,
  };
  const vector<float> b_data {
    -1, -2, -3, -4, -5, -6, -7, -8,
    -11, -22, -33, -44, -55, -66, -77, -88,
  };
  const vector<Shape> shapes {
    Shape({4, 2, 2}, 2),
    Shape({2, 4, 2}, 2),
    Shape({2, 2, 4}, 2),
    Shape({2, 2, 2, 2}, 2),
    Shape({2, 2, 2, 1, 2}, 2),
  };
  const vector<vector<float>> y_data {
    {1, 2, -1, -2, 3, 4, -3, -4, 5, 6, -5, -6, 7, 8, -7, -8,
      11, 22, -11, -22, 33, 44, -33, -44, 55, 66, -55, -66, 77, 88, -77, -88},
    {1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8, -5, -6, -7, -8,
      11, 22, 33, 44, -11, -22, -33, -44, 55, 66, 77, 88, -55, -66, -77, -88},
    {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
      11, 22, 33, 44, 55, 66, 77, 88, -11, -22, -33, -44, -55, -66, -77, -88},
  };
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), b_data);
    for (const unsigned i : {0, 1, 2, 3, 4}) {
      const Tensor y = concat({&a, &b}, i);
      EXPECT_EQ(shapes[i], y.shape());
      EXPECT_TRUE(vector_match(y_data[i < 2 ? i : 2], y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckConcatBatchBroadcast) {
  for (Device *dev : devices) {
    {
      const vector<float> y_data {
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        11, 11, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
      };
      const Tensor a = dev->new_tensor_by_vector(Shape({2, 1}, 2), {1, 1, 11, 11});
      const Tensor b = dev->new_tensor_by_vector({2, 2}, {2, 2, 2, 2});
      const Tensor c = dev->new_tensor_by_vector({2, 3}, {3, 3, 3, 3, 3, 3});
      const Tensor y = concat({&a, &b, &c}, 1);
      EXPECT_EQ(Shape({2, 6}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
    {
      const vector<float> y_data {
        1, 1, 1, 2, 2, 3, 1, 1, 1, 2, 2, 3,
        1, 1, 1, 22, 22, 33, 1, 1, 1, 22, 22, 33,
      };
      const Tensor a = dev->new_tensor_by_vector({3, 2}, {1, 1, 1, 1, 1, 1});
      const Tensor b = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {2, 2, 2, 2, 22, 22, 22, 22});
      const Tensor c = dev->new_tensor_by_vector(Shape({1, 2}, 2), {3, 3, 33, 33});
      const Tensor y = concat({&a, &b, &c}, 0);
      EXPECT_EQ(Shape({6, 2}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
    {
      const vector<float> y_data {1, 2, 3, 1, 2, 33, 1, 2, 333};
      const Tensor a = dev->new_tensor_by_vector({}, {1});
      const Tensor b = dev->new_tensor_by_vector({}, {2});
      const Tensor c = dev->new_tensor_by_vector(Shape({}, 3), {3, 33, 333});
      const Tensor y = concat({&a, &b, &c}, 0);
      EXPECT_EQ(Shape({3}, 3), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidConcat) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({1, 42}, 2), 0);
    const Tensor b = dev->new_tensor(Shape({2, 42}, 2), 0);
    const Tensor c = dev->new_tensor(Shape({1, 42}, 3), 0);
    const Tensor d = dev->new_tensor({2, 42}, 0);
    EXPECT_THROW(concat({}, 0), Error);
    EXPECT_NO_THROW(concat({&a, &b}, 0));
    EXPECT_THROW(concat({&a, &b}, 1), Error);
    EXPECT_THROW(concat({&a, &b}, 2), Error);
    EXPECT_THROW(concat({&a, &c}, 0), Error);
    EXPECT_THROW(concat({&a, &c}, 1), Error);
    EXPECT_THROW(concat({&a, &c}, 2), Error);
    EXPECT_THROW(concat({&b, &c}, 0), Error);
    EXPECT_THROW(concat({&b, &c}, 1), Error);
    EXPECT_THROW(concat({&b, &c}, 2), Error);
    EXPECT_NO_THROW(concat({&a, &d}, 0));
    EXPECT_THROW(concat({&a, &d}, 1), Error);
    EXPECT_THROW(concat({&a, &d}, 2), Error);
  }
}

TEST_F(TensorOpsTest, CheckReshape) {
  const vector<Shape> shapes {
    {6}, {1, 6}, {1, 1, 6}, {1, 1, 1, 6},
    {2, 3}, {2, 1, 3}, {1, 2, 3}, {2, 1, 1, 3}, {1, 2, 1, 3}, {1, 1, 2, 3},
    {3, 2}, {3, 1, 2}, {1, 3, 2}, {3, 1, 1, 2}, {1, 3, 1, 2}, {1, 1, 3, 2},
  };
  for (Device *dev : devices) {
    const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const Tensor a = dev->new_tensor_by_vector(Shape({6}, 2), data);
    for (const Shape &shape : shapes) {
      const Tensor y1 = reshape(a, shape);
      EXPECT_EQ(shape.resize_batch(2), y1.shape());
      EXPECT_TRUE(vector_match(data, y1.to_vector()));
      const Tensor y2 = reshape(a, shape.resize_batch(2));
      EXPECT_EQ(shape.resize_batch(2), y2.shape());
      EXPECT_TRUE(vector_match(data, y2.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidReshape) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor(Shape({6}, 2), 0);
    EXPECT_THROW(reshape(a, {7}), Error);
    EXPECT_THROW(reshape(a, Shape({6}, 3)), Error);
    EXPECT_THROW(reshape(a, Shape({7}, 3)), Error);
  }
}

TEST_F(TensorOpsTest, CheckFlatten) {
  const vector<Shape> shapes {
    {6}, {1, 6}, {1, 1, 6}, {1, 1, 1, 6},
    {2, 3}, {2, 1, 3}, {1, 2, 3}, {2, 1, 1, 3}, {1, 2, 1, 3}, {1, 1, 2, 3},
    {3, 2}, {3, 1, 2}, {1, 3, 2}, {3, 1, 1, 2}, {1, 3, 1, 2}, {1, 1, 3, 2},
  };
  for (Device *dev : devices) {
    const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (const Shape &shape : shapes) {
      const Tensor a = dev->new_tensor_by_vector(shape.resize_batch(2), data);
      const Tensor y = flatten(a);
      EXPECT_EQ(Shape({6}, 2), y.shape());
      EXPECT_TRUE(vector_match(data, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckDuplicate) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k + x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = x + k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckAddScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {10, 1};
  const vector<float> y_data {1010, 110, 20, 11, 1.1, 1.01, 1.001, 1.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    const Tensor y1 = k + x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = x + k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckAddScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {1};
    const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      const Tensor y1 = k + x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x + k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
  {
    const vector<float> x_data {1000, 100, 10, 1};
    const vector<float> k_data {10, 1};
    const vector<float> y_data {1010, 110, 20, 11, 1001, 101, 11, 2};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      const Tensor y1 = k + x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x + k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckAdd) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k - x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = x - k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSubtractScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {10, 1};
  const vector<float> y1_data {-990, -90, 0, 9, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {990, 90, 0, -9, -0.9, -0.99, -0.999, -0.9999};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    const Tensor y1 = k - x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = x - k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSubtractScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {1};
    const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
    const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      const Tensor y1 = k - x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x - k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
  {
    const vector<float> x_data {1000, 100, 10, 1};
    const vector<float> k_data {10, 1};
    const vector<float> y1_data {-990, -90, 0, 9, -999, -99, -9, 0};
    const vector<float> y2_data {990, 90, 0, -9, 999, 99, 9, 0};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      const Tensor y1 = k - x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x - k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckSubtract) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y1_data {1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006};
  const vector<float> y2_data {-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k * x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = x * k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckMultiplyScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {0.1, 10};
  const vector<float> y_data {100, 10, 1, 0.1, 1, 0.1, 0.01, 0.001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    const Tensor y1 = k * x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    const Tensor y2 = x * k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckMultiplyScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {10};
    const vector<float> y_data {10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      const Tensor y1 = k * x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x * k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
  {
    const vector<float> x_data {1000, 100, 10, 1};
    const vector<float> k_data {0.1, 10};
    const vector<float> y_data {100, 10, 1, 0.1, 10000, 1000, 100, 10};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      const Tensor y1 = k * x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x * k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckMultiply) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {0, 1, 2, 3, -4, -5, -6, -7};
  const vector<float> y_data {0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y1 = k / x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = x / k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDivideScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {10, 0.1};
  const vector<float> y1_data {0.01, 0.1, 1, 10, 1, 10, 100, 1000};
  const vector<float> y2_data {100, 10, 1, 0.1, 1, 0.1, 0.01, 0.001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    const Tensor y1 = k / x;
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = x / k;
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckDivideScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {10};
    const vector<float> y1_data {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
    const vector<float> y2_data {100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      const Tensor y1 = k / x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x / k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
  {
    const vector<float> x_data {1000, 100, 10, 1};
    const vector<float> k_data {10, 0.1};
    const vector<float> y1_data {0.01, 0.1, 1, 10, 0.0001, 0.001, 0.01, 0.1};
    const vector<float> y2_data {100, 10, 1, 0.1, 10000, 1000, 100, 10};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      const Tensor y1 = k / x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x / k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckDivide) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {1, 2, 3, 4, -5, -6, -7, -8};
  const vector<float> y1_data {
    1000, -50, 10.0/3, -0.25, -0.02, 0.01/6, -0.001/7, 1.25e-5,
  };
  const vector<float> y2_data {0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
  const vector<float> y2_data {1, 0.5, 1.0/3, 0.25, 1, 1, 1, 1};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
      const Tensor a = dev->new_tensor_by_vector(
          sa[i], vector<float>(sa[i].num_total_elements()));
      const Tensor b = dev->new_tensor_by_vector(
          sb[i], vector<float>(sb[i].num_total_elements()));
      EXPECT_THROW(a + b, Error);
      EXPECT_THROW(a - b, Error);
      EXPECT_THROW(a * b, Error);
      EXPECT_THROW(a / b, Error);
    }
  }
}

TEST_F(TensorOpsTest, CheckTranspose11) {
  for (Device *dev : devices) {
    const vector<float> x_data {42};
    const vector<float> y_data {42};
    const Tensor x = dev->new_tensor_by_vector({}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape(), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTransposeN1) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector({12}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({1, 12}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTranspose1N) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({1, 3}, 4), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3}, 4), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTransposeNN) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckTransposeMN) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckInvalidTranspose) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({2, 3, 4});
    EXPECT_THROW(transpose(x), Error);
  }
}

TEST_F(TensorOpsTest, CheckDotAA) {
  const vector<float> x_data {1, 2, 3, 4, 1, 0, 0, 1, 0, 2, 3, 0};
  const vector<float> y_data {7, 10, 15, 22, 1, 0, 0, 1, 6, 0, 0, 6};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
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
    const Tensor a = dev->new_tensor_by_vector({3, 4}, a_data);
    const Tensor b = dev->new_tensor_by_vector({4, 6}, b_data);
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
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
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
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
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
      EXPECT_THROW(dot(a, b), Error);
    }
    {
      // Not a scalar multiplication.
      const Tensor a = dev->new_tensor({});
      const Tensor b = dev->new_tensor({2, 3});
      EXPECT_THROW(dot(a, b), Error);
    }
    {
      const Tensor a = dev->new_tensor({2, 3, 4});
      const Tensor b = dev->new_tensor({4});
      EXPECT_THROW(dot(a, b), Error);
    }
    {
      const Tensor a = dev->new_tensor({1, 2});
      const Tensor b = dev->new_tensor({2, 3, 4});
      EXPECT_THROW(dot(a, b), Error);
    }
    {
      const Tensor a = dev->new_tensor({2, 3});
      const Tensor b = dev->new_tensor({2, 3});
      EXPECT_THROW(dot(a, b), Error);
    }
  }
}

TEST_F(TensorOpsTest, CheckSqrt) {
  const vector<float> x_data {
    0, 1, 2, 3, 4, 5,
    0, 1, 4, 9, 16, 25,
  };
  const vector<float> y_data {
    0, 1, 1.41421356, 1.73205041, 2, 2.23606798,
    0, 1, 2, 3, 4, 5,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = sqrt(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = step(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckPStep) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    .5, 1, 1, 1, 1, 1,
    .5, .5, .5, .5, .5, .5,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pstep(x, .5);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckReLU) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0, .5, 1, 2, 4, 8,
    0, 0, 0, 0, 0, 0,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = relu(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckPReLU) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0, .5, 1, 2, 4, 8,
    0, -.25, -.5, -1, -2, -4,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = prelu(x, .5);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSum) {
  const vector<float> x_data {
    1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
  };
  const vector<Shape> shape {
    Shape({1, 2, 2}, 2),
    Shape({2, 1, 2}, 2),
    Shape({2, 2}, 2),
    Shape({2, 2, 2}, 2),
  };
  const vector<vector<float>> y_data {
    {3, 7, 11, 15, -3, -7, -11, -15},
    {4, 6, 12, 14, -4, -6, -12, -14},
    {6, 8, 10, 12, -6, -8, -10, -12},
    {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), x_data);
    for (unsigned i = 0; i < 4; ++i) {
      const Tensor y = sum(x, i);
      EXPECT_EQ(shape[i], y.shape());
      EXPECT_TRUE(vector_match(y_data[i], y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckSum2) {
  const vector<unsigned> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const unsigned n : ns) {
      const Tensor x = dev->new_tensor({n}, 1);
      const Tensor y = sum(x, 0);
      EXPECT_EQ(Shape(), y.shape());
      EXPECT_TRUE(vector_match(vector<float>(1, n), y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckLogSumExp) {
  const vector<float> x_data {
    1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
  };
  const vector<Shape> shape {
    Shape({1, 2, 2}, 2),
    Shape({2, 1, 2}, 2),
    Shape({2, 2}, 2),
    Shape({2, 2, 2}, 2),
  };
  // NOTE(odashi): logsumexp(a, a + h) = a + log(1 + exp(h))
  const vector<vector<float>> y_data {
    {2.31326169, 4.31326169, 6.31326169, 8.31326169,
      -0.68673831, 2.68673831, -4.68673831, -6.68673831},
    {3.12692801, 4.12692801, 7.12692801, 8.12692801,
      -0.87307199, -1.87307199, -4.87307199, -5.87307199},
    {5.01814993, 6.01814993, 7.01814993, 8.01814993,
      -0.98185007, -1.98185007, -2.98185007, -3.98185007},
    {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), x_data);
    for (unsigned i = 0; i < 4; ++i) {
      const Tensor y = logsumexp(x, i);
      EXPECT_EQ(shape[i], y.shape());
      EXPECT_TRUE(vector_match(y_data[i], y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckLogSumExp2) {
  const vector<unsigned> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const unsigned n : ns) {
      for (const float k : {-5, -1, 0, 1, 5}) {
        const Tensor x = dev->new_tensor({n}, k);
        const Tensor y = logsumexp(x, 0);
        EXPECT_EQ(Shape(), y.shape());
        // TODO(odashi): 1e-3 might not be enough precision.
        EXPECT_TRUE(vector_near(
              vector<float>(1, k + std::log(n)), y.to_vector(), 1e-3));
    }
    }
  }
}

TEST_F(TensorOpsTest, CheckLogSoftmax) {
  const vector<float> x_data {
    1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
  };
  const vector<vector<float>> y_data {
    {-1.31326169, -0.31326169, -1.31326169, -0.31326169,
      -1.31326169, -0.31326169, -1.31326169, -0.31326169,
      -0.31326169, -1.31326169, -0.31326169, -1.31326169,
      -0.31326169, -1.31326169, -0.31326169, -1.31326169},
    {-2.12692801, -2.12692801, -0.12692801, -0.12692801,
      -2.12692801, -2.12692801, -0.12692801, -0.12692801,
      -0.12692801, -0.12692801, -2.12692801, -2.12692801,
      -0.12692801, -0.12692801, -2.12692801, -2.12692801},
    {-4.01814993, -4.01814993, -4.01814993, -4.01814993,
      -0.01814993, -0.01814993, -0.01814993, -0.01814993,
      -0.01814993, -0.01814993, -0.01814993, -0.01814993,
      -4.01814993, -4.01814993, -4.01814993, -4.01814993},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), x_data);
    for (unsigned i = 0; i < 4; ++i) {
      const Tensor y = log_softmax(x, i);
      EXPECT_EQ(Shape({2, 2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_near(y_data[i], y.to_vector(), 1e-6));
    }
  }
}

TEST_F(TensorOpsTest, CheckLogSoftmax2) {
  const vector<unsigned> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const unsigned n : ns) {
      for (const float k : {-5, -1, 0, 1, 5}) {
        const Tensor x = dev->new_tensor({n}, k);
        const Tensor y = log_softmax(x, 0);
        EXPECT_EQ(Shape({n}), y.shape());
        // TODO(odashi): 1e-3 might not be enough precision.
        EXPECT_TRUE(
            vector_near(vector<float>(n, -std::log(n)), y.to_vector(), 1e-3));
      }
    }
  }
}

TEST_F(TensorOpsTest, CheckSoftmax) {
  const vector<float> x_data {
    1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8,
  };
  const vector<vector<float>> y_data {
    {0.26894142, 0.73105858, 0.26894142, 0.73105858,
      0.26894142, 0.73105858, 0.26894142, 0.73105858,
      0.73105858, 0.26894142, 0.73105858, 0.26894142,
      0.73105858, 0.26894142, 0.73105858, 0.26894142},
    {0.11920292, 0.11920292, 0.88079708, 0.88079708,
      0.11920292, 0.11920292, 0.88079708, 0.88079708,
      0.88079708, 0.88079708, 0.11920292, 0.11920292,
      0.88079708, 0.88079708, 0.11920292, 0.11920292},
    {0.01798621, 0.01798621, 0.01798621, 0.01798621,
      0.98201379, 0.98201379, 0.98201379, 0.98201379,
      0.98201379, 0.98201379, 0.98201379, 0.98201379,
      0.01798621, 0.01798621, 0.01798621, 0.01798621},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), x_data);
    for (unsigned i = 0; i < 4; ++i) {
      const Tensor y = softmax(x, i);
      EXPECT_EQ(Shape({2, 2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_near(y_data[i], y.to_vector(), 1e-6));
    }
  }
}

TEST_F(TensorOpsTest, CheckSoftmax2) {
  const vector<unsigned> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const unsigned n : ns) {
      for (const float k : {-5, -1, 0, 1, 5}) {
        const Tensor x = dev->new_tensor({n}, k);
        const Tensor y = softmax(x, 0);
        EXPECT_EQ(Shape({n}), y.shape());
        EXPECT_TRUE(
            vector_near(vector<float>(n, 1./n), y.to_vector(), 1e-6));
      }
    }
  }
}

TEST_F(TensorOpsTest, CheckBroadcast) {
  struct TestCase {
    unsigned dim, size;
    Shape shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    {0, 1, {}, vector<float>(1, 1)},
    {0, 20, {20}, vector<float>(20, 1)},
    {1, 50, {1, 50}, vector<float>(50, 1)},
    {2, 100, {1, 1, 100}, vector<float>(100, 1)},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      const Tensor x = dev->new_tensor({}, 1);
      const Tensor y = broadcast(x, tc.dim, tc.size);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckBroadcast2) {
  struct TestCase {
    unsigned dim, size;
    Shape shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    {1, 1, Shape({2}, 3), {1, 2, 3, 4, 5, 6}},
    {2, 1, Shape({2}, 3), {1, 2, 3, 4, 5, 6}},
    {1, 2, Shape({2, 2}, 3), {1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}},
    {2, 2, Shape({2, 1, 2}, 3), {1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2}, 3), {1, 2, 3, 4, 5, 6});
      const Tensor y = broadcast(x, tc.dim, tc.size);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckBroadcast3) {
  struct TestCase {
    unsigned dim, size;
    Shape shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    {0, 1, Shape({1, 2, 1, 2}, 2),
      {1, 2, 3, 4, 5, 6, 7, 8}},
    {2, 1, Shape({1, 2, 1, 2}, 2),
      {1, 2, 3, 4, 5, 6, 7, 8}},
    {4, 1, Shape({1, 2, 1, 2}, 2),
      {1, 2, 3, 4, 5, 6, 7, 8}},
    {0, 2, Shape({2, 2, 1, 2}, 2),
      {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8}},
    {2, 2, Shape({1, 2, 2 ,2}, 2),
      {1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8}},
    {4, 2, Shape({1, 2, 1, 2, 2}, 2),
      {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({1, 2, 1, 2}, 2), {1, 2, 3, 4, 5, 6, 7, 8});
      const Tensor y = broadcast(x, tc.dim, tc.size);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckInvalidBroadcast) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({1, 2}, 0);
    EXPECT_THROW(broadcast(x, 0, 0), Error);
    EXPECT_THROW(broadcast(x, 1, 0), Error);
    EXPECT_THROW(broadcast(x, 1, 1), Error);
    EXPECT_THROW(broadcast(x, 1, 3), Error);
    EXPECT_THROW(broadcast(x, 2, 0), Error);
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
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), x_data);
    const Tensor y = batch_sum(x);
    EXPECT_EQ(Shape({2, 2, 2}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorOpsTest, CheckSoftmaxCrossEntropy) {
  const vector<vector<float>> x_data {
    {-1, 0, 1, 1, 0, 0, 0, 0, 1},
    {-1, 1, 0, 0, 0, 0, 1, 0, 1},
  };
  const vector<vector<float>> t_data {
    {1./3, 1./3, 1./3, .5, .25, .25, 0, 0, 1},
    {1./3, .5, 0, 1./3, .25, 0, 1./3, .25, 1},
  };
  const vector<vector<float>> y_data {
    {1.40760596, 1.05144471, 0.55144471},
    {1.40760596, 1.05144471, 0.55144471},
  };
  const vector<Shape> shape {{1, 3}, {3}};
  for (Device *dev : devices) {
    for (const unsigned dim : {0, 1}) {
      const Tensor x = dev->new_tensor_by_vector({3, 3}, x_data[dim]);
      const Tensor t = dev->new_tensor_by_vector({3, 3}, t_data[dim]);
      const Tensor y = softmax_cross_entropy(x, t, dim);
      EXPECT_EQ(shape[dim], y.shape());
      EXPECT_TRUE(vector_match(y_data[dim], y.to_vector()));
    }
  }
}

TEST_F(TensorOpsTest, CheckSoftmaxCrossEntropyBatchBroadcast) {
  struct TestCase {
    vector<float> x_data, t_data, y_data;
    Shape x_shape, t_shape, y_shape;
  };
  const vector<TestCase> test_cases {
    {{-1, 0, 1},
      {1, 0, 0, 0, 1, 0, 0, 0, 1},
      {2.40760596, 1.40760596, 0.40760596},
      {3}, Shape({3}, 3), Shape({}, 3)},
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1},
      {1, 0, 0},
      {2.40760596, 0.40760596, 1.40760596},
      Shape({3}, 3), {3}, Shape({}, 3)},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      const Tensor x = dev->new_tensor_by_vector(tc.x_shape, tc.x_data);
      const Tensor t = dev->new_tensor_by_vector(tc.t_shape, tc.t_data);
      const Tensor y = softmax_cross_entropy(x, t, 0);
      EXPECT_EQ(tc.y_shape, y.shape());
      EXPECT_TRUE(vector_match(tc.y_data, y.to_vector()));
    }
  }
}

}  // namespace tensor_ops
}  // namespace primitiv
