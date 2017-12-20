#include <primitiv/config.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/naive_device.h>
#include <primitiv/parameter.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match_ulps;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {
namespace functions {

class TensorForwardTest : public testing::Test {
protected:
  static vector<Device *> devices;

  static void SetUpTestCase() {
    test_utils::add_available_devices(devices);
  }

  static void TearDownTestCase() {
    for (Device *dev : devices) {
      delete dev;
    }
  }
};

vector<Device *> TensorForwardTest::devices;

TEST_F(TensorForwardTest, CheckInputByVector) {
  vector<float> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor y = input<Tensor>(Shape({2, 2}, 3), data, *dev);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_EQ(dev, &y.device());
    EXPECT_TRUE(vector_match(data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckInputByParameter) {
  vector<float> data {1, 2, 3, 4};
  for (Device *dev : devices) {
    Parameter param({2, 2}, data, *dev);
    const Tensor y = parameter<Tensor>(param);
    EXPECT_EQ(Shape({2, 2}), y.shape());
    EXPECT_EQ(dev, &y.device());
    EXPECT_TRUE(vector_match(data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckCopy) {
  vector<float> data(12);
  std::uint32_t i = 0;
  for (Device *dev : devices) {
    for (Device *dev2 : devices) {
      // Sets different (count-up) data to be copied every time.
      std::generate(data.begin(), data.end(), [&]() { i += 1; return i; });
      for (float x : data) std::cout << x << ' ';
      std::cout << std::endl;

      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), data);
      const Tensor y = copy(x, *dev2);
      EXPECT_EQ(Shape({2, 2}, 3), y.shape());
      EXPECT_TRUE(vector_match(data, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckInvalidCopy) {
  for (Device *dev : devices) {
    EXPECT_THROW(copy(Tensor(), *dev), Error);
  }
}

TEST_F(TensorForwardTest, CheckIdentity) {
  struct TestCase {
    std::uint32_t size;
    Shape shape;
    vector<float> values;
  };
  const vector<TestCase> test_cases {
    {1, {}, {1}},
    {2, {2, 2}, {1, 0, 0, 1}},
    {3, {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}},
    {4, {4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
  };
  for (Device *dev : devices) {
    Device::set_default(*dev);
    for (const TestCase &tc : test_cases) {
      const Tensor y = identity<Tensor>(tc.size);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckInvalidIdentity) {
  for (Device *dev : devices) {
    Device::set_default(*dev);
    EXPECT_THROW(identity<Tensor>(0), Error);
  }
}

TEST_F(TensorForwardTest, CheckPickNN) {
  struct TestCase {
    Shape x_shape;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
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
      for (std::uint32_t i = 0; i < tc.ids.size(); ++i) {
        if (i > 0) std::cerr << ',';
        std::cerr << tc.ids[i];
      }
      std::cerr << ']' << std::endl;
      vector<float> x_data(tc.x_shape.size());
      iota(x_data.begin(), x_data.end(), 0);
      const Tensor x = dev->new_tensor_by_vector(tc.x_shape, x_data);
      const Tensor y = pick(x, tc.ids, tc.dim);
      EXPECT_EQ(tc.y_shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckInvalidPick) {
  struct TestCase {
    std::uint32_t dim;
    vector<std::uint32_t> ids;
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
    const Tensor x = dev->new_tensor_by_constant(Shape({2, 2, 2}, 3), 0);
    for (const TestCase &tc : test_cases) {
      EXPECT_THROW(pick(x, tc.ids, tc.dim), Error);
    }
  }
}

TEST_F(TensorForwardTest, CheckSlice) {
  vector<float> x_data(3 * 3 * 2 * 4);
  std::iota(x_data.begin(), x_data.end(), 0);
  struct TestCase {
    std::uint32_t dim, lower, upper;
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

TEST_F(TensorForwardTest, CheckInvalidSlice) {
  struct TestCase { std::uint32_t dim, lower, upper; };
  const vector<TestCase> test_cases {
    {0, 0, 0}, {0, 1, 0}, {0, 0, 4}, {0, 3, 4},
    {1, 0, 0}, {1, 1, 0}, {1, 0, 4}, {1, 3, 4},
    {2, 0, 0}, {2, 1, 0}, {2, 0, 2}, {2, 1, 2},
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant({3, 3}, 3);
    for (const TestCase &tc : test_cases) {
      EXPECT_THROW(slice(x, tc.dim, tc.lower, tc.upper), Error);
    }
  }
}

TEST_F(TensorForwardTest, CheckConcatN_3x3) {
  const vector<float> y_data {
    1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
  };
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({1, 3}, {1, 1, 1});
    const Tensor b = dev->new_tensor_by_vector({2, 3}, {2, 3, 2, 3, 2, 3});
    const Tensor c = dev->new_tensor_by_vector({3, 3}, {4, 5, 6, 4, 5, 6, 4, 5, 6});
    const Tensor y1 = concat({a, b, c}, 0);
    const Tensor y2 = concat({&a, &b, &c}, 0);
    EXPECT_EQ(Shape({6, 3}), y1.shape());
    EXPECT_EQ(Shape({6, 3}), y2.shape());
    EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
    EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckConcat5x4) {
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
    for (const std::uint32_t i : {0, 1, 2}) {
      const Tensor y1 = concat({a, b, c, d}, i);
      const Tensor y2 = concat({&a, &b, &c, &d}, i);
      EXPECT_EQ(shapes[i], y1.shape());
      EXPECT_EQ(shapes[i], y2.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckConcat2_2_2x2) {
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
    for (const std::uint32_t i : {0, 1, 2, 3, 4}) {
      const Tensor y1 = concat({a, b}, i);
      const Tensor y2 = concat({&a, &b}, i);
      EXPECT_EQ(shapes[i], y1.shape());
      EXPECT_EQ(shapes[i], y2.shape());
      EXPECT_TRUE(vector_match(y_data[i < 2 ? i : 2], y1.to_vector()));
      EXPECT_TRUE(vector_match(y_data[i < 2 ? i : 2], y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckConcatBatchBroadcast) {
  for (Device *dev : devices) {
    {
      const vector<float> y_data {
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        11, 11, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
      };
      const Tensor a = dev->new_tensor_by_vector(Shape({2, 1}, 2), {1, 1, 11, 11});
      const Tensor b = dev->new_tensor_by_vector({2, 2}, {2, 2, 2, 2});
      const Tensor c = dev->new_tensor_by_vector({2, 3}, {3, 3, 3, 3, 3, 3});
      const Tensor y1 = concat({a, b, c}, 1);
      const Tensor y2 = concat({&a, &b, &c}, 1);
      EXPECT_EQ(Shape({2, 6}, 2), y1.shape());
      EXPECT_EQ(Shape({2, 6}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
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
      const Tensor y1 = concat({a, b, c}, 0);
      const Tensor y2 = concat({&a, &b, &c}, 0);
      EXPECT_EQ(Shape({6, 2}, 2), y1.shape());
      EXPECT_EQ(Shape({6, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const vector<float> y_data {1, 2, 3, 1, 2, 33, 1, 2, 333};
      const Tensor a = dev->new_tensor_by_vector({}, {1});
      const Tensor b = dev->new_tensor_by_vector({}, {2});
      const Tensor c = dev->new_tensor_by_vector(Shape({}, 3), {3, 33, 333});
      const Tensor y1 = concat({a, b, c}, 0);
      const Tensor y2 = concat({&a, &b, &c}, 0);
      EXPECT_EQ(Shape({3}, 3), y1.shape());
      EXPECT_EQ(Shape({3}, 3), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckInvalidConcat) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant(Shape({1, 42}, 2), 0);
    const Tensor b = dev->new_tensor_by_constant(Shape({2, 42}, 2), 0);
    const Tensor c = dev->new_tensor_by_constant(Shape({1, 42}, 3), 0);
    const Tensor d = dev->new_tensor_by_constant({2, 42}, 0);

    // NOTE(odashi): Now these lines generate compile errors.
    //EXPECT_THROW(concat({}, 0), Error);

    EXPECT_NO_THROW(concat({a, b}, 0));
    EXPECT_THROW(concat({a, b}, 1), Error);
    EXPECT_THROW(concat({a, b}, 2), Error);
    EXPECT_THROW(concat({a, c}, 0), Error);
    EXPECT_THROW(concat({a, c}, 1), Error);
    EXPECT_THROW(concat({a, c}, 2), Error);
    EXPECT_THROW(concat({b, c}, 0), Error);
    EXPECT_THROW(concat({b, c}, 1), Error);
    EXPECT_THROW(concat({b, c}, 2), Error);
    EXPECT_NO_THROW(concat({a, d}, 0));
    EXPECT_THROW(concat({a, d}, 1), Error);
    EXPECT_THROW(concat({a, d}, 2), Error);

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

TEST_F(TensorForwardTest, CheckReshape) {
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

TEST_F(TensorForwardTest, CheckInvalidReshape) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant(Shape({6}, 2), 0);
    EXPECT_THROW(reshape(a, {7}), Error);
    EXPECT_THROW(reshape(a, Shape({6}, 3)), Error);
    EXPECT_THROW(reshape(a, Shape({7}, 3)), Error);
  }
}

TEST_F(TensorForwardTest, CheckFlatten) {
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

TEST_F(TensorForwardTest, CheckDuplicate) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y = +x;
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(x_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckNegate) {
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

TEST_F(TensorForwardTest, CheckAddConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    {
      const Tensor y1 = add(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = add(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k + x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x + k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckAddScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {10, 1};
  const vector<float> y_data {1010, 110, 20, 11, 1.1, 1.01, 1.001, 1.0001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    {
      const Tensor y1 = add(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = add(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k + x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x + k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckAddScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {1};
    const vector<float> y_data {1001, 101, 11, 2, 1.1, 1.01, 1.001, 1.0001};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      {
        const Tensor y1 = add(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = add(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k + x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = x + k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
    }
  }
  {
    const vector<float> x_data {1000, 100, 10, 1};
    const vector<float> k_data {10, 1};
    const vector<float> y_data {1010, 110, 20, 11, 1001, 101, 11, 2};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      {
        const Tensor y1 = add(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = add(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k + x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = x + k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
    }
  }
}

TEST_F(TensorForwardTest, CheckAdd) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y_data {1000, 200, 30, 4, 0.5, 0.06, 0.007, 0.0008};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = add(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = add(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a + b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = b + a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckAddBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y_data {0, 1, 2, 3, 4, 5, 6, 7};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = add(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = add(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a + b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = b + a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSubtractConst) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const float k = 1;
  const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    {
      const Tensor y1 = subtract(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = subtract(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k - x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x - k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSubtractScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {10, 1};
  const vector<float> y1_data {-990, -90, 0, 9, 0.9, 0.99, 0.999, 0.9999};
  const vector<float> y2_data {990, 90, 0, -9, -0.9, -0.99, -0.999, -0.9999};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    {
      const Tensor y1 = subtract(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = subtract(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k - x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x - k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSubtractScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {1};
    const vector<float> y1_data {-999, -99, -9, 0, 0.9, 0.99, 0.999, 0.9999};
    const vector<float> y2_data {999, 99, 9, 0, -0.9, -0.99, -0.999, -0.9999};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      {
        const Tensor y1 = subtract(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = subtract(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k - x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = x - k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
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
      {
        const Tensor y1 = subtract(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = subtract(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k - x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = x - k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
    }
  }
}

TEST_F(TensorForwardTest, CheckSubtract) {
  const vector<float> a_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> b_data {   0, 100, 20, 3, 0.4, 0.05, 0.006, 0.0007};
  const vector<float> y1_data {1000, 0, -10, -2, -0.3, -0.04, -0.005, -0.0006};
  const vector<float> y2_data {-1000, 0, 10, 2, 0.3, 0.04, 0.005, 0.0006};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = subtract(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = subtract(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a - b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = b - a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSubtractBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {0, 0, 0, 0, 4, 4, 4, 4};
  const vector<float> y1_data {0, 1, 2, 3, -4, -3, -2, -1};
  const vector<float> y2_data {0, -1, -2, -3, 4, 3, 2, 1};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = subtract(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = subtract(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a - b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = b - a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckMultiplyConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y_data {10000, -1000, 100, -10, 1, -0.1, 0.01, -0.001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    {
      const Tensor y1 = multiply(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = multiply(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k * x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x * k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckMultiplyScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {0.1, 10};
  const vector<float> y_data {100, 10, 1, 0.1, 1, 0.1, 0.01, 0.001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    {
      const Tensor y1 = multiply(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = multiply(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k * x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = x * k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckMultiplyScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {10};
    const vector<float> y_data {10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      {
        const Tensor y1 = multiply(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = multiply(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k * x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = x * k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
    }
  }
  {
    const vector<float> x_data {1000, 100, 10, 1};
    const vector<float> k_data {0.1, 10};
    const vector<float> y_data {100, 10, 1, 0.1, 10000, 1000, 100, 10};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      {
        const Tensor y1 = multiply(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = multiply(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k * x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
        const Tensor y2 = x * k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
      }
    }
  }
}

TEST_F(TensorForwardTest, CheckMultiply) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {0, 1, 2, 3, -4, -5, -6, -7};
  const vector<float> y_data {0, -100, 20, -3, -0.4, 0.05, -0.006, 0.0007};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = multiply(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = multiply(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a * b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = b * a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckMultiplyBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 0, 1, 2, 3};
  const vector<float> y_data {0, 1, 2, 3, 0, 1, 4, 9};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = multiply(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = multiply(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a * b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y_data, y1.to_vector()));
      const Tensor y2 = b * a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckDivideConst) {
  const vector<float> x_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const float k = 10;
  const vector<float> y1_data {0.01, -0.1, 1, -10, 100, -1000, 10000, -100000};
  const vector<float> y2_data {
    100, -10, 1, -0.1, 0.01, -0.001, 0.0001, -0.00001,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    {
      const Tensor y1 = divide(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = divide(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k / x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x / k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckDivideScalar) {
  const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
  const vector<float> k_data {10, 0.1};
  const vector<float> y1_data {0.01, 0.1, 1, 10, 1, 10, 100, 1000};
  const vector<float> y2_data {100, 10, 1, 0.1, 1, 0.1, 0.01, 0.001};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    {
      const Tensor y1 = divide(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = divide(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = k / x;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = x / k;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckDivideScalarBatchBroadcast) {
  {
    const vector<float> x_data {1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001};
    const vector<float> k_data {10};
    const vector<float> y1_data {0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
    const vector<float> y2_data {100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001};
    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      {
        const Tensor y1 = divide(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = divide(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k / x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = x / k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
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
      {
        const Tensor y1 = divide(k, x);
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = divide(x, k);
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
      {
        const Tensor y1 = k / x;
        EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
        EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
        const Tensor y2 = x / k;
        EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
        EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
      }
    }
  }
}

TEST_F(TensorForwardTest, CheckDivide) {
  const vector<float> a_data {1000, -100, 10, -1, 0.1, -0.01, 0.001, -0.0001};
  const vector<float> b_data {1, 2, 3, 4, -5, -6, -7, -8};
  const vector<float> y1_data {
    1000, -50, 10.0/3, -0.25, -0.02, 0.01/6, -0.001/7, 1.25e-5,
  };
  const vector<float> y2_data {0.001, -0.02, 0.3, -4, -50, 600, -7000, 80000};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = divide(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = divide(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a / b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = b / a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckDivideBatchBroadcast) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {1, 1, 1, 1, 1, 2, 3, 4};
  const vector<float> y1_data {1, 2, 3, 4, 1, 1, 1, 1};
  const vector<float> y2_data {1, 0.5, 1.0/3, 0.25, 1, 1, 1, 1};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    {
      const Tensor y1 = divide(a, b);
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = divide(b, a);
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
    {
      const Tensor y1 = a / b;
      EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
      EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
      const Tensor y2 = b / a;
      EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
      EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckPowConstR) {
  const vector<float> x_data {1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4};
  const float k = 3;
  const vector<float> y_data {1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12};

  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y = pow(x, k);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowConstL) {
  const vector<float> x_data {3, 2, 1, 0, -1, -2, -3, -4};
  const float k = 3;
  const vector<float> y_data {27, 9, 3, 1, 1./3, 1./9, 1./27, 1./81};

  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor y = pow(k, x);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowScalarR) {
  const vector<float> x_data {1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4};
  const vector<float> k_data {3, -3};
  const vector<float> y_data {1e9, 1e6, 1e3, 1e0, 1e3, 1e6, 1e9, 1e12};

  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    const Tensor y = pow(x, k);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowScalarL) {
  const vector<float> x_data {3, 2, 1, 0, -1, -2, -3, -4};
  const vector<float> k_data {2, 3};
  const vector<float> y_data {8, 4, 2, 1, 1./3, 1./9, 1./27, 1./81};

  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
    const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
    const Tensor y = pow(k, x);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowScalarRBatchBroadcast) {
  {
    const vector<float> x_data {1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4};
    const vector<float> k_data {3};
    const vector<float> y_data {1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9, 1e-12};

    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      const Tensor y = pow(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
  {
    const vector<float> x_data {1e3, 1e2, 1e1, 1e0};
    const vector<float> k_data {3, -3};
    const vector<float> y_data {1e9, 1e6, 1e3, 1e0, 1e-9, 1e-6, 1e-3, 1e0};

    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      const Tensor y = pow(x, k);
      EXPECT_EQ(Shape({2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckPowScalarLBatchBroadcast) {
  {
    const vector<float> x_data {3, 2, 1, 0, -1, -2, -3, -4};
    const vector<float> k_data {3};
    const vector<float> y_data {27, 9, 3, 1, 1./3, 1./9, 1./27, 1./81};

    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_data);
      const Tensor k = dev->new_tensor_by_vector({}, k_data);
      const Tensor y = pow(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
  {
    const vector<float> x_data {3, 2, 1, 0};
    const vector<float> k_data {2, 3};
    const vector<float> y_data {8, 4, 2, 1, 27, 9, 3, 1};

    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector({2, 2}, x_data);
      const Tensor k = dev->new_tensor_by_vector(Shape({}, 2), k_data);
      const Tensor y = pow(k, x);
      EXPECT_EQ(Shape({2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckPow) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y1_data {0, 1, 4, 9, 0, 1, 8, 27};
  const vector<float> y2_data {1, 2, 4, 8, 1, 3, 9, 27};

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    const Tensor y1 = pow(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = pow(b, a);
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowBatchBroadcast) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y1_data {0, 1, 4, 9, 0, 1, 8, 27};
  const vector<float> y2_data {1, 2, 4, 8, 1, 3, 9, 27};

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    const Tensor y1 = pow(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y1.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    const Tensor y2 = pow(b, a);
    EXPECT_EQ(Shape({2, 2}, 2), y2.shape());
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckInvalidArithmeticOps) {
  const vector<Shape> sa {
    Shape({2, 2}, 2), Shape({2, 2}, 2), Shape({2, 2}, 2),
  };
  const vector<Shape> sb {
    Shape({2, 2}, 3), Shape({3, 3}, 2), Shape({3, 3}, 3),
  };
  for (Device *dev : devices) {
    for (std::uint32_t i = 0; i < sa.size(); ++i) {
      const Tensor a = dev->new_tensor_by_vector(
          sa[i], vector<float>(sa[i].size()));
      const Tensor b = dev->new_tensor_by_vector(
          sb[i], vector<float>(sb[i].size()));
      EXPECT_THROW(add(a, b), Error);
      EXPECT_THROW(a + b, Error);
      EXPECT_THROW(subtract(a, b), Error);
      EXPECT_THROW(a - b, Error);
      EXPECT_THROW(multiply(a, b), Error);
      EXPECT_THROW(a * b, Error);
      EXPECT_THROW(divide(a, b), Error);
      EXPECT_THROW(a / b, Error);
      EXPECT_THROW(pow(a, b), Error);
    }
  }
}

TEST_F(TensorForwardTest, CheckTranspose11) {
  for (Device *dev : devices) {
    const vector<float> x_data {42};
    const vector<float> y_data {42};
    const Tensor x = dev->new_tensor_by_vector({}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape(), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckTransposeN1) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector({12}, x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({1, 12}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckTranspose1N) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({1, 3}, 4), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3}, 4), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckTransposeNN) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckTransposeMN) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = transpose(x);
    EXPECT_EQ(Shape({3, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckInvalidTranspose) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant({2, 3, 4}, 0);
    EXPECT_THROW(transpose(x), Error);
  }
}

TEST_F(TensorForwardTest, CheckMatMulAA) {
  const vector<float> x_data {1, 2, 3, 4, 1, 0, 0, 1, 0, 2, 3, 0};
  const vector<float> y_data {7, 10, 15, 22, 1, 0, 0, 1, 6, 0, 0, 6};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
    const Tensor y = matmul(x, x);
    EXPECT_EQ(Shape({2, 2}, 3), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckMatMulAB) {
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
    const Tensor y = matmul(a, b);
    EXPECT_EQ(Shape({3, 6}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckMatMulBatchBroadcast1N) {
  const vector<float> a_data {10, 1000, 1, 100};
  const vector<float> b_data {1, 2, 3, 4, 5, 6, 7, 8};
  const vector<float> y_data {12, 1200, 34, 3400, 56, 5600, 78, 7800};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_data);
    const Tensor y = matmul(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckMatMulBatchBroadcastN1) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8};
  const vector<float> b_data {10, 1, 1000, 100};
  const vector<float> y_data {13, 24, 1300, 2400, 57, 68, 5700, 6800};
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_data);
    const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
    const Tensor y = matmul(a, b);
    EXPECT_EQ(Shape({2, 2}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckMatMulLarge) {
  const std::uint32_t N = 123;
  vector<float> a_data(N * N);
  vector<float> b_data(N * N);
  vector<float> y1_data(N * N);
  vector<float> y2_data(N * N);
  std::uint32_t k = 0;
  for (std::uint32_t i = 0; i < N; ++i) {
    k += i * i;
  }
  for (std::uint32_t i = 0; i < N; ++i) {
    for (std::uint32_t j = 0; j < N; ++j) {
      a_data[i + j * N] = i;
      b_data[i + j * N] = j;
      y1_data[i + j * N] = N * i * j;
      y2_data[i + j * N] = k;
    }
  }
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({N, N}), a_data);
    const Tensor b = dev->new_tensor_by_vector({N, N}, b_data);
    const Tensor y1 = matmul(a, b);
    const Tensor y2 = matmul(b, a);
    EXPECT_EQ(Shape({N, N}), y1.shape());
    EXPECT_EQ(Shape({N, N}), y2.shape());
    EXPECT_TRUE(vector_match(y1_data, y1.to_vector()));
    EXPECT_TRUE(vector_match(y2_data, y2.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckInvalidMatMul) {
  for (Device *dev : devices) {
    {
      // Not a scalar multiplication.
      const Tensor a = dev->new_tensor_by_constant({2, 3}, 0);
      const Tensor b = dev->new_tensor_by_constant({}, 0);
      EXPECT_THROW(matmul(a, b), Error);
    }
    {
      // Not a scalar multiplication.
      const Tensor a = dev->new_tensor_by_constant({}, 0);
      const Tensor b = dev->new_tensor_by_constant({2, 3}, 0);
      EXPECT_THROW(matmul(a, b), Error);
    }
    {
      const Tensor a = dev->new_tensor_by_constant({2, 3, 4}, 0);
      const Tensor b = dev->new_tensor_by_constant({4}, 0);
      EXPECT_THROW(matmul(a, b), Error);
    }
    {
      const Tensor a = dev->new_tensor_by_constant({1, 2}, 0);
      const Tensor b = dev->new_tensor_by_constant({2, 3, 4}, 0);
      EXPECT_THROW(matmul(a, b), Error);
    }
    {
      const Tensor a = dev->new_tensor_by_constant({2, 3}, 0);
      const Tensor b = dev->new_tensor_by_constant({2, 3}, 0);
      EXPECT_THROW(matmul(a, b), Error);
    }
  }
}

TEST_F(TensorForwardTest, CheckSqrt) {
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

TEST_F(TensorForwardTest, CheckExp) {
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

TEST_F(TensorForwardTest, CheckLog) {
  const vector<float> x_data {
    0.01, .5, 1, 2, 4, 8,
    0.01, .5, 1, 2, 4, 8,
  };
  const vector<float> y_data {
    -4.60517019, -0.69314718, 0, 0.69314718, 1.38629436, 2.07944154,
    -4.60517019, -0.69314718, 0, 0.69314718, 1.38629436, 2.07944154,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = log(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowNPositive) {
  const vector<float> x_data {
    0.01, .5, 1, 2, 4, 8,
    -0.01, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0.000001, 0.125, 1, 8, 64, 512,
    -0.000001, -0.125, -1, -8, -64, -512,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pown(x, 3);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowNNegative) {
  const vector<float> x_data {
    0.01, .5, 1, 2, 4, 8,
    -0.01, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    1000000, 8, 1, 0.125, 0.015625, 0.001953125,
    -1000000, -8, -1, -0.125, -0.015625, -0.001953125,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pown(x, -3);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowNUpperBound) {
  const vector<float> x_data {
    1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, 1, -1,
  };
  const vector<float> y_data {
    1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, 1, -1,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pown(x, 0x7fffffff);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowNLowerBound) {
  const vector<float> x_data {
    1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, 1, -1,
  };
  const vector<float> y_data {
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pown(x, 0x80000000);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowNPositiveConvergence) {
  const vector<float> x_data {
    0.9999999, -0.9999999, 0.9999999, -0.9999999, 0.9999999, -0.9999999,
    0.9999999, -0.9999999, 0.9999999, -0.9999999, 0.9999999, -0.9999999,
  };
  const vector<float> y_data {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pown(x, 0x7fffffff);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPowNNegativeConvergence) {
  const vector<float> x_data {
    1.000001, -1.000001, 1.000001, -1.000001, 1.000001, -1.000001,
    1.000001, -1.000001, 1.000001, -1.000001, 1.000001, -1.000001,
  };
  const vector<float> y_data {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = pown(x, 0x80000000);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckTanh) {
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

TEST_F(TensorForwardTest, CheckSigmoid) {
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
    EXPECT_TRUE(vector_match_ulps(y_data, y.to_vector(), 6));
  }
}

TEST_F(TensorForwardTest, CheckSoftplus) {
  const vector<float> x_data {
    0, .5, 1, 2, 3, 4,
    0, -.5, -1, -2, -3, -4,
  };
  const vector<float> y_data {
    .69314718, .97407698, 1.3132617, 2.1269280, 3.0485874, 4.0181499,
    .69314718, .47407698, .31326169, .12692801, .048587352, .018149928,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = softplus(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_near(y_data, y.to_vector(), 1e-6));
  }
}

TEST_F(TensorForwardTest, CheckSin) {
  const vector<float> x_data {
    0, .5, 1, 2, 3, 4,
    0, -.5, -1, -2, -3, -4,
  };
  const vector<float> y_data {
    0, .47942554, .84147098, .90929743, .14112001, -.75680250,
    0, -.47942554, -.84147098, -.90929743, -.14112001, .75680250,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = sin(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckCos) {
  const vector<float> x_data {
    0, .5, 1, 2, 3, 4,
    0, -.5, -1, -2, -3, -4,
  };
  const vector<float> y_data {
    1, .87758256, .54030231, -.41614684, -.98999250, -.65364362,
    1, .87758256, .54030231, -.41614684, -.98999250, -.65364362,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = cos(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckTan) {
  const vector<float> x_data {
    0, .5, 1, 2, 3, 4,
    0, -.5, -1, -2, -3, -4,
  };
  const vector<float> y_data {
    0, .54630249, 1.5574077, -2.1850399, -.14254654, 1.1578213,
    0, -.54630249, -1.5574077, 2.1850399, .14254654, -1.1578213,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = tan(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckReLU) {
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

TEST_F(TensorForwardTest, CheckLReLU) {
  const vector<float> x_data {
    0, .5, 1, 2, 4, 8,
    0, -.5, -1, -2, -4, -8,
  };
  const vector<float> y_data {
    0, .5, 1, 2, 4, 8,
    0, -.005, -.01, -.02, -.04, -.08,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = lrelu(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckPReLU) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const vector<float> x_data {
        0, .5, 1, 2, 4, 8,
        0, -.5, -1, -2, -4, -8,
      };
      const vector<float> y_data {
        0, .5, 1, 2, 4, 8,
        0, -.5f * k, -k, -2 * k, -4 * k, -8 * k,
      };
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
      const Tensor y = prelu(x, k);
      EXPECT_EQ(Shape({2, 3}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckELU) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const vector<float> x_data {
        0, .5, 1, 2, 4, 8,
        0, -.5, -1, -2, -4, -8,
      };
      const vector<float> y_data {
        0, .5, 1, 2, 4, 8,
        0, -.39346934f * k, -.63212056f * k,
        -.86466472f * k, -.98168436f * k, -.99966454f * k,
      };
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
      const Tensor y = elu(x, k);
      EXPECT_EQ(Shape({2, 3}, 2), y.shape());
      EXPECT_TRUE(vector_match(y_data, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSum) {
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
    for (std::uint32_t i = 0; i < 4; ++i) {
      const Tensor y = sum(x, i);
      EXPECT_EQ(shape[i], y.shape());
      EXPECT_TRUE(vector_match(y_data[i], y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSum2) {
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      const Tensor x = dev->new_tensor_by_constant({n}, 1);
      const Tensor y = sum(x, 0);
      EXPECT_EQ(Shape(), y.shape());
      EXPECT_TRUE(vector_match(vector<float>(1, n), y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckLogSumExp) {
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
    for (std::uint32_t i = 0; i < 4; ++i) {
      const Tensor y = logsumexp(x, i);
      EXPECT_EQ(shape[i], y.shape());
      EXPECT_TRUE(vector_match(y_data[i], y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckLogSumExp2) {
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      for (const float k : {-5, -1, 0, 1, 5}) {
        const Tensor x = dev->new_tensor_by_constant({n}, k);
        const Tensor y = logsumexp(x, 0);
        EXPECT_EQ(Shape(), y.shape());
        // TODO(odashi): 1e-3 might not be enough precision.
        EXPECT_TRUE(vector_near(
              vector<float>(1, k + std::log(n)), y.to_vector(), 1e-3));
    }
    }
  }
}

TEST_F(TensorForwardTest, CheckLogSoftmax) {
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
    for (std::uint32_t i = 0; i < 4; ++i) {
      const Tensor y = log_softmax(x, i);
      EXPECT_EQ(Shape({2, 2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_near(y_data[i], y.to_vector(), 1e-6));
    }
  }
}

TEST_F(TensorForwardTest, CheckLogSoftmax2) {
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      for (const float k : {-5, -1, 0, 1, 5}) {
        const Tensor x = dev->new_tensor_by_constant({n}, k);
        const Tensor y = log_softmax(x, 0);
        EXPECT_EQ(Shape({n}), y.shape());
        // TODO(odashi): 1e-3 might not be enough precision.
        EXPECT_TRUE(
            vector_near(vector<float>(n, -std::log(n)), y.to_vector(), 1e-3));
      }
    }
  }
}

TEST_F(TensorForwardTest, CheckSoftmax) {
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
    for (std::uint32_t i = 0; i < 4; ++i) {
      const Tensor y = softmax(x, i);
      EXPECT_EQ(Shape({2, 2, 2}, 2), y.shape());
      EXPECT_TRUE(vector_near(y_data[i], y.to_vector(), 1e-6));
    }
  }
}

TEST_F(TensorForwardTest, CheckSoftmax2) {
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537,
  };
  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      for (const float k : {-5, -1, 0, 1, 5}) {
        const Tensor x = dev->new_tensor_by_constant({n}, k);
        const Tensor y = softmax(x, 0);
        EXPECT_EQ(Shape({n}), y.shape());
        EXPECT_TRUE(
            vector_near(vector<float>(n, 1./n), y.to_vector(), 1e-6));
      }
    }
  }
}

TEST_F(TensorForwardTest, CheckBroadcast) {
  struct TestCase {
    std::uint32_t dim, size;
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
      const Tensor x = dev->new_tensor_by_constant({}, 1);
      const Tensor y = broadcast(x, tc.dim, tc.size);
      EXPECT_EQ(tc.shape, y.shape());
      EXPECT_TRUE(vector_match(tc.values, y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckBroadcast2) {
  struct TestCase {
    std::uint32_t dim, size;
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

TEST_F(TensorForwardTest, CheckBroadcast3) {
  struct TestCase {
    std::uint32_t dim, size;
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

TEST_F(TensorForwardTest, CheckInvalidBroadcast) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant({1, 2}, 0);
    EXPECT_THROW(broadcast(x, 0, 0), Error);
    EXPECT_THROW(broadcast(x, 1, 0), Error);
    EXPECT_THROW(broadcast(x, 1, 1), Error);
    EXPECT_THROW(broadcast(x, 1, 3), Error);
    EXPECT_THROW(broadcast(x, 2, 0), Error);
  }
}

TEST_F(TensorForwardTest, CheckBatchSum) {
  const vector<float> x_data {
    1, 2, 3, 4, 5, 6, 7, 8,
    -2, -4, -6, -8, -10, -12, -14, -16,
  };
  const vector<float> y_data {
    -1, -2, -3, -4, -5, -6, -7, -8,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 2, 2}, 2), x_data);
    const Tensor y = batch::sum(x);
    EXPECT_EQ(Shape({2, 2, 2}), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

TEST_F(TensorForwardTest, CheckSoftmaxCrossEntropy) {
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
    for (const std::uint32_t dim : {0, 1}) {
      const Tensor x = dev->new_tensor_by_vector({3, 3}, x_data[dim]);
      const Tensor t = dev->new_tensor_by_vector({3, 3}, t_data[dim]);
      const Tensor y = softmax_cross_entropy(x, t, dim);
      EXPECT_EQ(shape[dim], y.shape());
      EXPECT_TRUE(vector_match(y_data[dim], y.to_vector()));
    }
  }
}

TEST_F(TensorForwardTest, CheckSoftmaxCrossEntropyBatchBroadcast) {
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

TEST_F(TensorForwardTest, CheckInvalidSoftmaxCrossEntropy) {
  for (Device *dev : devices) {
    {
      const Tensor x = dev->new_tensor_by_constant({2, 2}, .5);
      const Tensor t = dev->new_tensor_by_constant({2, 3}, .5);
      EXPECT_THROW(softmax_cross_entropy(x, t, 0), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 1), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 2), Error);
    }
    {
      const Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), .5);
      const Tensor t = dev->new_tensor_by_constant(Shape({2, 3}, 3), .5);
      EXPECT_THROW(softmax_cross_entropy(x, t, 0), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 1), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 2), Error);
    }
  }
}

TEST_F(TensorForwardTest, CheckSparseSoftmaxCrossEntropy) {
  struct TestCase {
    vector<float> x_data;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    Shape x_shape, y_shape;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    // Testing 1-1 operations with 0/1/2 dimensions.
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1},
      0, {0}, {3, 3}, {1, 3},
      {2.40760596, 0.40760596, 1.40760596}},
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1},
      1, {1}, {3, 3}, {3},
      {0.40760596, 2.40760596, 1.40760596}},
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1},
      2, {0}, {3, 3}, {3, 3},
      {0, 0, 0, 0, 0, 0, 0, 0, 0}},
    // Testing N-N operation.
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1, -2, 0, 2, 2, -2, 0, 0, 2, -2},
      0, {0, 1}, Shape({3, 3}, 2), Shape({1, 3}, 2),
      {2.40760596, 0.40760596, 1.40760596,
        2.14293163, 4.14293163, 0.14293163}},
    // Testing N-1 operation.
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1, -2, 0, 2, 2, -2, 0, 0, 2, -2},
      0, {0}, Shape({3, 3}, 2), Shape({1, 3}, 2),
      {2.40760596, 0.40760596, 1.40760596,
        4.14293163, 0.14293163, 2.14293163}},
    // Testing 1-N operation.
    {{-1, 0, 1, 1, -1, 0, 0, 1, -1},
      0, {0, 1}, {3, 3}, Shape({1, 3}, 2),
      {2.40760596, 0.40760596, 1.40760596,
        1.40760596, 2.40760596, 0.40760596}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      const Tensor x = dev->new_tensor_by_vector(tc.x_shape, tc.x_data);
      const Tensor y = softmax_cross_entropy(x, tc.ids, tc.dim);
      EXPECT_EQ(tc.y_shape, y.shape());
      EXPECT_TRUE(vector_near(tc.y_data, y.to_vector(), 1e-6));
    }
  }
}

TEST_F(TensorForwardTest, CheckInvalidSparseSoftmaxCrossEntropy) {
  for (Device *dev : devices) {
    {
      const Tensor x = dev->new_tensor_by_constant({2, 2}, .5);
      const vector<std::uint32_t> t {2};
      EXPECT_THROW(softmax_cross_entropy(x, t, 0), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 1), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 2), Error);
    }
    {
      const Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), .5);
      const vector<std::uint32_t> t {0, 0, 0};
      EXPECT_THROW(softmax_cross_entropy(x, t, 0), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 1), Error);
      EXPECT_THROW(softmax_cross_entropy(x, t, 2), Error);
    }
  }
}

TEST_F(TensorForwardTest, CheckStopGradient) {
  const vector<float> x_data {
    0, .5, 1, 2, 3, 4,
    0, -.5, -1, -2, -3, -4,
  };
  const vector<float> y_data = x_data;
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 2), x_data);
    const Tensor y = stop_gradient(x);
    EXPECT_EQ(Shape({2, 3}, 2), y.shape());
    EXPECT_TRUE(vector_match(y_data, y.to_vector()));
  }
}

}  // namespace functions
}  // namespace primitiv
