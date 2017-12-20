#include <primitiv/config.h>

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/naive_device.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match_ulps;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class TensorBackwardTest : public testing::Test {
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

vector<Device *> TensorBackwardTest::devices;

TEST_F(TensorBackwardTest, CheckSliceNN_1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    for (std::uint32_t i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      dev->slice_bw(b, i, 0, a);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSliceNN_2) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 2, 2, 3, 3};
  struct TestCase {
    Shape shape;
    std::uint32_t dim, offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {Shape({1, 2}, 3), 0, 0, {1, 1, 3, 3, 2, 1, 4, 3, 3, 1, 5, 3}},
    {Shape({1, 2}, 3), 0, 1, {0, 2, 2, 4, 0, 3, 2, 5, 0, 4, 2, 6}},
    {Shape({2}, 3), 1, 0, {1, 2, 2, 3, 2, 3, 2, 3, 3, 4, 2, 3}},
    {Shape({2}, 3), 1, 1, {0, 1, 3, 4, 0, 1, 4, 5, 0, 1, 5, 6}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.shape, b_data);
      dev->slice_bw(b, tc.dim, tc.offset, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSlice1N_1) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {6, 7, 8, 9};
  for (Device *dev : devices) {
    for (std::uint32_t i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      dev->slice_bw(b, i, 0, a);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSlice1N_2) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 2, 2, 3, 3};
  struct TestCase {
    Shape shape;
    std::uint32_t dim, offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {Shape({1, 2}, 3), 0, 0, {6, 1, 8, 3}},
    {Shape({1, 2}, 3), 0, 1, {0, 7, 2, 9}},
    {Shape({2}, 3), 1, 0, {6, 7, 2, 3}},
    {Shape({2}, 3), 1, 1, {0, 1, 8, 9}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.shape, b_data);
      dev->slice_bw(b, tc.dim, tc.offset, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSliceN1_1) {
  const vector<float> a_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> b_data {-1, -2, -3, -4};
  const vector<float> y_data {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  for (Device *dev : devices) {
    for (std::uint32_t i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
      dev->slice_bw(b, i, 0, a);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSliceN1_2) {
  const vector<float> a_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> b_data {-1, -2};
  struct TestCase {
    Shape shape;
    std::uint32_t dim, offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {{1, 2}, 0, 0, {0, 2, 1, 4, 1, 3, 2, 5, 2, 4, 3, 6}},
    {{1, 2}, 0, 1, {1, 1, 3, 2, 2, 2, 4, 3, 3, 3, 5, 4}},
    {{2}, 1, 0, {0, 0, 3, 4, 1, 1, 4, 5, 2, 2, 5, 6}},
    {{2}, 1, 1, {1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.shape, b_data);
      dev->slice_bw(b, tc.dim, tc.offset, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckInvalidSlice) {
  struct TestCase {
    Shape a_shape, b_shape;
    std::uint32_t dim, offset;
    bool ok;
  };
  vector<TestCase> test_cases {
    {Shape({}, 2), Shape({}, 3), 0, 0, false},
    {Shape({42}, 2), Shape({42}, 3), 0, 0, false},
    {{}, {}, 0, 0, true},
    {{}, {}, 0, 1, false},
    {{42}, {}, 0, 41, true},
    {{42}, {}, 0, 42, false},
    {{42}, {42}, 0, 0, true},
    {{42}, {42}, 0, 1, false},
    {{42}, {43}, 0, 0, false},
    {{42}, {4, 42}, 0, 0, false},
    {{42}, {4, 2, 42}, 0, 0, false},
    {{4, 4}, {2, 2}, 0, 0, false},
    {{4, 4}, {2, 4}, 0, 2, true},
    {{4, 4}, {2, 4}, 0, 3, false},
    {{4, 4}, {4, 2}, 1, 2, true},
    {{4, 4}, {4, 2}, 1, 3, false},
    {{4, 4}, {4, 4}, 2, 0, true},
    {{4, 4}, {4, 4}, 2, 1, false},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_constant(tc.a_shape, 0);
      const Tensor b = dev->new_tensor_by_constant(tc.b_shape, 0);
      if (tc.ok) {
        EXPECT_NO_THROW(dev->slice_bw(b, tc.dim, tc.offset, a));
      } else {
        EXPECT_THROW(dev->slice_bw(b, tc.dim, tc.offset, a), Error);
      }
    }
  }
}

TEST_F(TensorBackwardTest, CheckCopyAndSlice) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    for (std::uint32_t i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

      const Tensor copied = a;

      dev->slice_bw(b, i, 0, a);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
      EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPickNN) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {0, 0, 0},
      {1, 1, 3, 3, 2, 1, 4, 3, 3, 1, 5, 3}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {1, 1, 1},
      {0, 2, 2, 4, 0, 3, 2, 5, 0, 4, 2, 6}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {0, 1, 0},
      {1, 1, 3, 3, 0, 3, 2, 5, 3, 1, 5, 3}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {1, 0, 1},
      {0, 2, 2, 4, 2, 1, 4, 3, 0, 4, 2, 6}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {0, 0, 0},
      {1, 2, 2, 3, 2, 3, 2, 3, 3, 4, 2, 3}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {1, 1, 1},
      {0, 1, 3, 4, 0, 1, 4, 5, 0, 1, 5, 6}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {0, 1, 0},
      {1, 2, 2, 3, 0, 1, 4, 5, 3, 4, 2, 3}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {1, 0, 1},
      {0, 1, 3, 4, 2, 3, 2, 3, 0, 1, 5, 6}},
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, 2, {0, 0, 0},
      {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.b_shape, tc.b_data);
      dev->pick_bw(b, tc.ids, tc.dim, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPickN1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {0},
      {1, 1, 3, 3, 2, 1, 4, 3, 3, 1, 5, 3}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {1},
      {0, 2, 2, 4, 0, 3, 2, 5, 0, 4, 2, 6}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {0},
      {1, 2, 2, 3, 2, 3, 2, 3, 3, 4, 2, 3}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {1},
      {0, 1, 3, 4, 0, 1, 4, 5, 0, 1, 5, 6}},
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, 2, {0},
      {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.b_shape, tc.b_data);
      dev->pick_bw(b, tc.ids, tc.dim, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPick1N) {
  const vector<float> a_data {0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {0, 0, 0},
      {6, 1, 8, 3}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {1, 1, 1},
      {0, 7, 2, 9}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {0, 1, 0},
      {4, 3, 6, 5}},
    {Shape({1, 2}, 3), {1, 1, 2, 2, 3, 3}, 0, {1, 0, 1},
      {2, 5, 4, 7}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {0, 0, 0},
      {6, 7, 2, 3}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {1, 1, 1},
      {0, 1, 8, 9}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {0, 1, 0},
      {4, 5, 4, 5}},
    {Shape({2}, 3), {1, 1, 2, 2, 3, 3}, 1, {1, 0, 1},
      {2, 3, 6, 7}},
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, 2, {0, 0, 0},
      {6, 7, 8, 9}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.b_shape, tc.b_data);
      dev->pick_bw(b, tc.ids, tc.dim, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckInvalidPick) {
  struct TestCase {
    Shape a_shape, b_shape;
    std::uint32_t dim;
    vector<std::uint32_t> ids;
  };
  vector<TestCase> test_cases {
    // Out-of-range IDs.
    {{}, {}, 0, {1}},
    {{}, Shape({}, 3), 0, {0, 0, 1}},
    {Shape({}, 3), Shape({}, 3), 0, {1}},
    {Shape({}, 3), Shape({}, 3), 0, {0, 0, 1}},
    // Batch size mismatched.
    {{}, {}, 0, {}},
    {{}, {}, 0, {0, 0, 0}},
    {Shape({}, 3), {}, 0, {}},
    {Shape({}, 3), {}, 0, {0, 0, 0}},
    // Shape mismatched.
    {{2}, {3}, 0, {0}},
    {{2}, Shape({3}, 3), 0, {0, 0, 0}},
    {Shape({2}, 3), Shape({3}, 3), 0, {0}},
    {Shape({2}, 3), Shape({3}, 3), 0, {0, 0, 0}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_constant(tc.a_shape, 0);
      const Tensor b = dev->new_tensor_by_constant(tc.b_shape, 0);
      EXPECT_THROW(dev->pick_bw(b, tc.ids, tc.dim, a), Error);
    }
  }
}

TEST_F(TensorBackwardTest, CheckCopyAndPick) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

    const Tensor copied = a;

    dev->pick_bw(b, {0, 0, 0}, 2, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSqrt) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0.01, 1, 4, 9, 0.01, 1, 4, 9});
    const Tensor y = dev->sqrt_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->sqrt_bw(x, y, gy, gx);
    const vector<float> gx_val {5, -.5, .5, -1./3, 10, -1, .25, -1./6};
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckExp) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->exp_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->exp_bw(x, y, gy, gx);
    const vector<float> gx_val {
      1, -2.7182818, 14.778112, -40.171074,
      2, -.73575888, .13533528, -.049787068,
    };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckLog) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0.01, 1, 2, 3, 0.01, 1, 2, 3});
    const Tensor y = dev->log_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->log_bw(x, y, gy, gx);
    const vector<float> gx_val { 100, -1, 1, -2./3, 200, -2, .5, -1./3 };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckTanh) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->tanh_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->tanh_bw(x, y, gy, gx);
    const vector<float> gx_val {
      1, -.41997434, .14130165, -.019732074,
      2, -.83994868, .070650825, -.0098660372,
    };
    EXPECT_TRUE(vector_near(gx_val, gx.to_vector(), 1e-6));
  }
}

TEST_F(TensorBackwardTest, CheckSigmoid) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->sigmoid_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->sigmoid_bw(x, y, gy, gx);
    const vector<float> gx_val {
      .25, -.19661193, .20998717, -.090353319,
      .5, -.39322387, .10499359, -.045176660,
    };
    EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), 6));
  }
}

TEST_F(TensorBackwardTest, CheckSoftplus) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->softplus_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->softplus_bw(x, y, gy, gx);
    const vector<float> gx_val {
      .5, -.73105858, 1.7615942, -1.9051483,
      1, -.53788284, .11920292, -.047425873,
    };
    EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), 6));
  }
}

TEST_F(TensorBackwardTest, CheckSin) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->sin_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->sin_bw(x, y, gy, gx);
    const vector<float> gx_val {
      1, -.54030231, -.83229367, 1.9799850,
      2, -1.0806046, -.41614684, .98999250,
    };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckCos) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->cos_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->cos_bw(x, y, gy, gx);
    const vector<float> gx_val {
      0, .84147098, -1.8185949, .28224002,
      0, -1.6829420, .90929743, -.14112001,
    };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckTan) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->tan_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->tan_bw(x, y, gy, gx);
    const vector<float> gx_val {
      1, -3.4255188, 11.548798, -2.0406390,
      2, -6.8510376, 5.7743992, -1.0203195,
    };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckAddConst) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->add_const_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->add_const_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        1, -1, 2, -2, 2, -2, 1, -1,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSubtractConstR) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->subtract_const_r_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->subtract_const_r_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        1, -1, 2, -2, 2, -2, 1, -1,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckSubtractConstL) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->subtract_const_l_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->subtract_const_l_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        -1, 1, -2, 2, -2, 2, -1, 1,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckMultiplyConst) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->multiply_const_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->multiply_const_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        k, -k, 2 *k, -2 * k, 2 * k, -2 * k, k, -k,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckDivideConstR) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->divide_const_r_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->divide_const_r_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        1 / k, -1 / k, 2 / k, -2 / k, 2 / k, -2 / k, 1 / k, -1 / k,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckDivideConstL) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {.1, 1, 2, 3, -.1, -1, -2, -3});
      const Tensor y = dev->divide_const_l_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->divide_const_l_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
       -100 * k, k, -k / 2, 2 * k / 9, -200 * k, 2 * k, -k / 4, k / 9,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPowConstR) {
  const vector<float> x_val {1, 2, 4, 8, 16, 32, 64, 128};
  const vector<float> ks {2., 1., .5, 0, -.5, -1., -2., -4.};
  auto fgx = [](float x, float k) { return k * std::pow(x, k - 1); };

  for (const float k : ks) {
    vector<float> gx_val;
    for (float x : x_val) { gx_val.emplace_back(fgx(x, k)); }

    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_val);
      const Tensor y = dev->pow_const_r_fw(x, k);
      const Tensor gy = dev->new_tensor_by_constant(y.shape(), 1);
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->pow_const_r_bw(x, y, gy, k, gx);
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPowConstL) {
  const vector<float> x_val {2., 1., .5, 0, -.5, -1., -2., -4.};
  const vector<float> ks {1, 2, 4, 8, 16, 32, 64, 128};
  auto fgx = [](float x, float k) { return std::log(k) * std::pow(k, x); };

  for (const float k : ks) {
    vector<float> gx_val;
    for (float x : x_val) { gx_val.emplace_back(fgx(x, k)); }

    for (Device *dev : devices) {
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), x_val);
      const Tensor y = dev->pow_const_l_fw(x, k);
      const Tensor gy = dev->new_tensor_by_constant(y.shape(), 1);
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->pow_const_l_bw(x, y, gy, k, gx);
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPReLU) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->prelu_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->prelu_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        k, -1, 2, -2, 2 * k, -2 * k, k, -k,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckELU) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev : devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->elu_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
      dev->elu_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        k, -1, 2, -2,
        2 * k, -7.3575888e-01f * k, 1.3533528e-01f * k, -4.9787068e-02f * k,
      };
      EXPECT_TRUE(vector_near(gx_val, gx.to_vector(), 1e-5));
    }
  }
}

TEST_F(TensorBackwardTest, CheckTranspose) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({3, 4}, 2), 0);
    const Tensor y = dev->transpose_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({4, 3}, 2), {
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        });
    Tensor gx = dev->new_tensor_by_constant(Shape({3, 4}, 2), 0);
    dev->transpose_bw(x, y, gy, gx);
    const vector<float> gx_val {
      0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
      12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23,
    };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckAdd11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant({2}, 0);
    const Tensor b = dev->new_tensor_by_constant({2}, 0);
    const Tensor y = dev->add_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector({2}, {1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->add_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1};
    const vector<float> gb_val {1, -1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckAddNN) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor b = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor y = dev->add_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->add_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 2, -2};
    const vector<float> gb_val {1, -1, 2, -2};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckAdd1N) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant({2}, 0);
    const Tensor b = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor y = dev->add_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->add_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {3, -3};
    const vector<float> gb_val {1, -1, 2, -2};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckAddN1) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor b = dev->new_tensor_by_constant({2}, 0);
    const Tensor y = dev->add_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->add_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 2, -2};
    const vector<float> gb_val {3, -3};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSubtract11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant({2}, 0);
    const Tensor b = dev->new_tensor_by_constant({2}, 0);
    const Tensor y = dev->subtract_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector({2}, {1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->subtract_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1};
    const vector<float> gb_val {-1, 1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSubtractNN) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor b = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor y = dev->subtract_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->subtract_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 2, -2};
    const vector<float> gb_val {-1, 1, -2, 2};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSubtract1N) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant({2}, 0);
    const Tensor b = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor y = dev->subtract_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->subtract_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {3, -3};
    const vector<float> gb_val {-1, 1, -2, 2};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSubtractN1) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_constant(Shape({2}, 2), 0);
    const Tensor b = dev->new_tensor_by_constant({2}, 0);
    const Tensor y = dev->subtract_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->subtract_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 2, -2};
    const vector<float> gb_val {-3, 3};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMultiply11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2}, {1, 10});
    const Tensor b = dev->new_tensor_by_vector({2}, {10, 1});
    const Tensor y = dev->multiply_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector({2}, {1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->multiply_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {10, -1};
    const vector<float> gb_val {1, -10};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMultiplyNN) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2}, 2), {1, 10, -1, -10});
    const Tensor b = dev->new_tensor_by_vector(Shape({2}, 2), {10, 1, -10, -1});
    const Tensor y = dev->multiply_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->multiply_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {10, -1, -20, 2};
    const vector<float> gb_val {1, -10, -2, 20};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMultiply1N) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2}, {1, 10});
    const Tensor b = dev->new_tensor_by_vector(Shape({2}, 2), {10, 1, -10, -1});
    const Tensor y = dev->multiply_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->multiply_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {-10, 1};
    const vector<float> gb_val {1, -10, 2, -20};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMultiplyN1) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2}, 2), {1, 10, -1, -10});
    const Tensor b = dev->new_tensor_by_vector({2}, {10, 1});
    const Tensor y = dev->multiply_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->multiply_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {10, -1, 20, -2};
    const vector<float> gb_val {-1, 10};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckDivide11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2}, {1, 10});
    const Tensor b = dev->new_tensor_by_vector({2}, {10, 1});
    const Tensor y = dev->divide_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector({2}, {1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->divide_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {.1, -1};
    const vector<float> gb_val {-.01, 10};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckDivideNN) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2}, 2), {1, 10, -1, -10});
    const Tensor b = dev->new_tensor_by_vector(Shape({2}, 2), {10, 1, -10, -1});
    const Tensor y = dev->divide_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->divide_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {.1, -1, -.2, 2};
    const vector<float> gb_val {-.01, 10, .02, -20};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckDivide1N) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2}, {1, 10});
    const Tensor b = dev->new_tensor_by_vector(Shape({2}, 2), {10, 1, -10, -1});
    const Tensor y = dev->divide_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->divide_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {-.1, 1};
    const vector<float> gb_val {-.01, 10, -.02, 20};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckDivideN1) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2}, 2), {1, 10, -1, -10});
    const Tensor b = dev->new_tensor_by_vector({2}, {10, 1});
    const Tensor y = dev->divide_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2}, 2), {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->divide_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {.1, -1, .2, -2};
    const vector<float> gb_val {.01, -10};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPow11) {
  auto fga = [](float a, float b) { return b * std::pow(a, b - 1); };
  auto fgb = [](float a, float b) { return std::log(a) * std::pow(a, b); };

  const vector<float> a_val {1, 2, 4, 8};
  const vector<float> b_val {2, 1, 0, -1};
  vector<float> ga_val, gb_val;
  for (std::size_t i = 0; i < a_val.size(); ++i) {
    ga_val.emplace_back(fga(a_val[i], b_val[i]));
    gb_val.emplace_back(fgb(a_val[i], b_val[i]));
  }

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_val);
    const Tensor b = dev->new_tensor_by_vector({2, 2}, b_val);
    const Tensor y = dev->pow_fw(a, b);
    const Tensor gy = dev->new_tensor_by_constant(y.shape(), 1);
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->pow_bw(a, b, y, gy, ga, gb);
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPowNN) {
  auto fga = [](float a, float b) { return b * std::pow(a, b - 1); };
  auto fgb = [](float a, float b) { return std::log(a) * std::pow(a, b); };

  const vector<float> a_val {1, 2, 4, 8, 16, 32, 64, 128};
  const vector<float> b_val {3, 2, 1, 0, -1, -2, -3, -4};
  vector<float> ga_val, gb_val;
  for (std::size_t i = 0; i < a_val.size(); ++i) {
    ga_val.emplace_back(fga(a_val[i], b_val[i]));
    gb_val.emplace_back(fgb(a_val[i], b_val[i]));
  }

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_val);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_val);
    const Tensor y = dev->pow_fw(a, b);
    const Tensor gy = dev->new_tensor_by_constant(y.shape(), 1);
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->pow_bw(a, b, y, gy, ga, gb);
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPow1N) {
  auto fga = [](float a, float b) { return b * std::pow(a, b - 1); };
  auto fgb = [](float a, float b) { return std::log(a) * std::pow(a, b); };

  const vector<float> a_val {1, 2, 4, 8};
  const vector<float> b_val {3, 2, 1, 0, -1, -2, -3, -4};
  vector<float> ga_val(a_val.size(), 0), gb_val(b_val.size(), 0);
  for (std::size_t ib = 0; ib < b_val.size(); ++ib) {
    const std::size_t ia = ib % a_val.size();
    ga_val[ia] += fga(a_val[ia], b_val[ib]);
    gb_val[ib] += fgb(a_val[ia], b_val[ib]);
  }

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, a_val);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 2), b_val);
    const Tensor y = dev->pow_fw(a, b);
    const Tensor gy = dev->new_tensor_by_constant(y.shape(), 1);
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->pow_bw(a, b, y, gy, ga, gb);
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPowN1) {
  auto fga = [](float a, float b) { return b * std::pow(a, b - 1); };
  auto fgb = [](float a, float b) { return std::log(a) * std::pow(a, b); };

  const vector<float> a_val {1, 2, 4, 8, 16, 32, 64, 128};
  const vector<float> b_val {2, 1, 0, -1};
  vector<float> ga_val(a_val.size(), 0), gb_val(b_val.size(), 0);
  for (std::size_t ia = 0; ia < a_val.size(); ++ia) {
    const std::size_t ib = ia % b_val.size();
    ga_val[ia] += fga(a_val[ia], b_val[ib]);
    gb_val[ib] += fgb(a_val[ia], b_val[ib]);
  }

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 2), a_val);
    const Tensor b = dev->new_tensor_by_vector({2, 2}, b_val);
    const Tensor y = dev->pow_fw(a, b);
    const Tensor gy = dev->new_tensor_by_constant(y.shape(), 1);
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->pow_bw(a, b, y, gy, ga, gb);
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMul11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, {1, 2, 3, 4});
    const Tensor b = dev->new_tensor_by_vector({2, 2}, {1, 0, 0, 2});
    const Tensor y = dev->matmul_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector({2, 2}, {1, -1, 2, -2});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->matmul_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4};
    const vector<float> gb_val {-1, -1, -2, -2};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMulNN) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 2, 3, 4, -1, -2, -3, -4});
    const Tensor b = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 0, 0, 2, 0, 1, 2, 0});
    const Tensor y = dev->matmul_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->matmul_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4, 2, -2, 2, -2};
    const vector<float> gb_val {-1, -1, -2, -2, 2, 2, 1, 1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMul1N) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, {1, 2, 3, 4});
    const Tensor b = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 0, 0, 2, 0, 1, 2, 0});
    const Tensor y = dev->matmul_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->matmul_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {3, -3, 6, -6};
    const vector<float> gb_val {-1, -1, -2, -2, -2, -2, -1, -1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMulN1) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 2, 3, 4, -1, -2, -3, -4});
    const Tensor b = dev->new_tensor_by_vector({2, 2}, {1, 0, 0, 2});
    const Tensor y = dev->matmul_fw(a, b);
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor_by_constant(a.shape(), 0);
    Tensor gb = dev->new_tensor_by_constant(b.shape(), 0);
    dev->matmul_bw(a, b, y, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4, 2, -2, 2, -2};
    const vector<float> gb_val {1, 1, -1, -1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

}  // namespace primitiv
