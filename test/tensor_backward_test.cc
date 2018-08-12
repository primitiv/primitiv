#include <primitiv/config.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/tensor.h>

#include <test_utils.h>

using std::vector;
using test_utils::get_default_ulps;
using test_utils::make_iota_vector;
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

TEST_F(TensorBackwardTest, CheckAbs) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0.01, .0, 1, 2.5, -0.01, -.0, -1, -2.5});
    const Tensor y = dev->abs_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor_by_constant(x.shape(), 0);
    dev->abs_bw(x, y, gy, gx);
    const vector<float> gx_val { 1, 0, 2, -2, -2, 0, -1, 1 };
    EXPECT_TRUE(vector_match_ulps(
          gx_val, gx.to_vector(), get_default_ulps(*dev)));
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
    EXPECT_TRUE(vector_match_ulps(
          gx_val, gx.to_vector(), get_default_ulps(*dev)));
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
    EXPECT_TRUE(
        vector_match_ulps(gx_val, gx.to_vector(), get_default_ulps(*dev)));
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
    EXPECT_TRUE(vector_match_ulps(
          gx_val, gx.to_vector(), get_default_ulps(*dev)));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 150000
      : 96;
    EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 32768
      : dev_type == DeviceType::EIGEN ? 6
      : dev_type == DeviceType::OPENCL ? 6
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::EIGEN ? 6
      : dev_type == DeviceType::OPENCL ? 6
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
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
    EXPECT_TRUE(vector_match_ulps(
          gx_val, gx.to_vector(), get_default_ulps(*dev)));
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
    EXPECT_TRUE(vector_match_ulps(
          gx_val, gx.to_vector(), get_default_ulps(*dev)));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
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
      EXPECT_TRUE(vector_match_ulps(
            gx_val, gx.to_vector(), get_default_ulps(*dev)));
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
      EXPECT_TRUE(vector_match_ulps(
            gx_val, gx.to_vector(), get_default_ulps(*dev)));
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

      const auto dev_type = dev->type();
      const std::uint32_t ulps
        = dev_type == DeviceType::CUDA16 ? 8192
        : get_default_ulps(*dev);
      EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPowConstR) {
  const vector<float> x_val {1, 2, 4, 8, 16, 32, 64, 128};
  const vector<float> ks {1., .5, .25, 0, -.125, -.25, -.5, -.1};
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

      const auto dev_type = dev->type();
      const std::uint32_t ulps
        = dev_type == DeviceType::CUDA16 ? 8192
        : get_default_ulps(*dev);
      EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPowConstL) {
  const vector<float> x_val {1., .5, .25, 0, -.125, -.25, -.5, -1.};
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

      const auto dev_type = dev->type();
      const std::uint32_t ulps
        = dev_type == DeviceType::CUDA16 ? 8192
        : get_default_ulps(*dev);
      EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
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
      EXPECT_TRUE(vector_match_ulps(
            gx_val, gx.to_vector(), get_default_ulps(*dev)));
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

      const auto dev_type = dev->type();
      const std::uint32_t ulps
        = dev_type == DeviceType::CUDA16 ? 75000
        : 12;
      EXPECT_TRUE(vector_match_ulps(gx_val, gx.to_vector(), ulps));
    }
  }
}

TEST_F(TensorBackwardTest, CheckMaxDims) {
  const vector<float> x_data = {
    0, 1, 2, 6, 7, 8, 3, 4, 5, -3, -4, -5, 0, -1, -2, -6, -7, -8,
  };
  const vector<vector<float>> y_data = {
    {2, 8, 5, -3, 0, -6},
    {6, 7, 8, 0, -1, -2},
    {0, 1, 2, 6, 7, 8, 3, 4, 5, -3, -4, -5, 0, -1, -2, -6, -7, -8},
  };
  const vector<vector<float>> gy_data = {
    {1, 2, 6, 5, 3, 4},
    {-1, 1, -2, 2, -3, 3},
    {0, 1, 0, -1, 0, 1, 0, -1, 2, 1, 0, -1, 0, 1, 2, 3, 4, 6},
  };
  const vector<vector<float>> expected = {
    {1, 1, 2, 1, 1, 3, 1, 1, 7, 6, 1, 1, 4, 1, 1, -5, 1, 1},
    {1, 1, 1, 0, 2, -1, 1, 1, 1, 1, 1, 1, 3, -2, 4, 1, 1, 1},
    {1, 2, 1, 0, 1, 2, 1, 0, 3, 2, 1, 0, 1, 2, 3, 4, 5, 7},
  };

  for (Device *dev : devices) {
    for (const std::uint32_t i : {0u, 1u, 2u}) {
      try {
        const Shape r({3, 3}, 2);
        const Shape s = r.resize_dim(i, 1);
        const Tensor x = dev->new_tensor_by_vector(r, x_data);
        const Tensor y = dev->new_tensor_by_vector(s, y_data[i]);
        const Tensor gy = dev->new_tensor_by_vector(s, gy_data[i]);
        Tensor gx = dev->new_tensor_by_constant(r, 1);
        dev->max_bw(x, y, gy, i, gx);
        EXPECT_TRUE(vector_match(expected[i], gx.to_vector()));
      } IGNORE_NOT_IMPLEMENTED
    }
  }
}

TEST_F(TensorBackwardTest, CheckMaxLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(vbkaisetsu):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }

      vector<float> x_data(n);
      vector<float> y_data = {static_cast<float>(n - 1)};
      vector<float> gy_data = {1};
      std::iota(begin(x_data), end(x_data), 0);
      std::shuffle(begin(x_data), end(x_data), rng);
      const auto it = std::find(begin(x_data), end(x_data), n - 1);
      const std::uint32_t pos = std::distance(begin(x_data), it);
      vector<float> expected(n, 1);
      expected[pos] = 2;
      const Tensor x = dev->new_tensor_by_vector({n}, x_data);
      const Tensor y = dev->new_tensor_by_vector({1}, y_data);
      const Tensor gy = dev->new_tensor_by_vector({1}, gy_data);
      Tensor gx = dev->new_tensor_by_constant({n}, 1);
      dev->max_bw(x, y, gy, 0, gx);
      EXPECT_TRUE(vector_match(expected, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckMaxMultipleLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(vbkaisetsu):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }

      vector<float> x_data(n);
      vector<float> y_data = {static_cast<float>(n - 1)};
      vector<float> gy_data = {1};
      std::iota(begin(x_data), end(x_data), 0);
      // NOTE(vbkaisetsu):
      // Generates a tensor that has some duplicated maximum values.
      for (std::uint32_t i = 0; i < 10 && i < n; ++i) {
        x_data[i] = n - 1;
      }
      std::shuffle(begin(x_data), end(x_data), rng);
      const auto it = std::find(begin(x_data), end(x_data), n - 1);
      const std::uint32_t pos = std::distance(begin(x_data), it);
      vector<float> expected(n, 1);
      expected[pos] = 2;
      const Tensor x = dev->new_tensor_by_vector({n}, x_data);
      const Tensor y = dev->new_tensor_by_vector({1}, y_data);
      const Tensor gy = dev->new_tensor_by_vector({1}, gy_data);
      Tensor gx = dev->new_tensor_by_constant({n}, 1);
      dev->max_bw(x, y, gy, 0, gx);
      EXPECT_TRUE(vector_match(expected, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckMinDims) {
  const vector<float> x_data = {
    3, 4, 5, 0, 1, 2, 6, 7, 8, 0, -1, -2, -6, -7, -8, -3, -4, -5,
  };
  const vector<vector<float>> y_data = {
    {3, 0, 6, -2, -8, -5},
    {0, 1, 2, -6, -7, -8},
    {3, 4, 5, 0, 1, 2, 6, 7, 8, 0, -1, -2, -6, -7, -8, -3, -4, -5},
  };
  const vector<vector<float>> gy_data = {
    {1, 2, 6, 5, 3, 4},
    {-1, 1, -2, 2, -3, 3},
    {0, 1, 0, -1, 0, 1, 0, -1, 2, 1, 0, -1, 0, 1, 2, 3, 4, 6},
  };
  const vector<vector<float>> expected = {
    {2, 1, 1, 3, 1, 1, 7, 1, 1, 1, 1, 6, 1, 1, 4, 1, 1, 5},
    {1, 1, 1, 0, 2, -1, 1, 1, 1, 1, 1, 1, 3, -2, -7, 1, 1, 1},
    {1, 2, 1, 0, 1, 2, 1, 0, 3, 2, 1, 0, 1, 2, 3, 4, 5, 7},
  };

  for (Device *dev : devices) {
    for (const std::uint32_t i : {0u, 1u, 2u}) {
      try {
        const Shape r({3, 3}, 2);
        const Shape s = r.resize_dim(i, 1);
        const Tensor x = dev->new_tensor_by_vector(r, x_data);
        const Tensor y = dev->new_tensor_by_vector(s, y_data[i]);
        const Tensor gy = dev->new_tensor_by_vector(s, gy_data[i]);
        Tensor gx = dev->new_tensor_by_constant(r, 1);
        dev->min_bw(x, y, gy, i, gx);
        EXPECT_TRUE(vector_match(expected[i], gx.to_vector()));
      } IGNORE_NOT_IMPLEMENTED
    }
  }
}

TEST_F(TensorBackwardTest, CheckMinLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(vbkaisetsu):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }

      vector<float> x_data(n);
      vector<float> y_data = {0};
      vector<float> gy_data = {1};
      std::iota(begin(x_data), end(x_data), 0);
      std::shuffle(begin(x_data), end(x_data), rng);
      const auto it = std::find(begin(x_data), end(x_data), 0);
      const std::uint32_t pos = std::distance(begin(x_data), it);
      vector<float> expected(n, 1);
      expected[pos] = 2;
      const Tensor x = dev->new_tensor_by_vector({n}, x_data);
      const Tensor y = dev->new_tensor_by_vector({1}, y_data);
      const Tensor gy = dev->new_tensor_by_vector({1}, gy_data);
      Tensor gx = dev->new_tensor_by_constant({n}, 1);
      dev->min_bw(x, y, gy, 0, gx);
      EXPECT_TRUE(vector_match(expected, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckMinMultipleLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(vbkaisetsu):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }

      vector<float> x_data(n);
      vector<float> y_data = {0};
      vector<float> gy_data = {1};
      std::iota(begin(x_data), end(x_data), 0);
      // NOTE(vbkaisetsu):
      // Generates a tensor that has some duplicated minimum values.
      for (std::uint32_t i = 0; i < 10 && i < n; ++i) {
        x_data[i] = 0;
      }
      std::shuffle(begin(x_data), end(x_data), rng);
      const auto it = std::find(begin(x_data), end(x_data), 0);
      const std::uint32_t pos = std::distance(begin(x_data), it);
      vector<float> expected(n, 1);
      expected[pos] = 2;
      const Tensor x = dev->new_tensor_by_vector({n}, x_data);
      const Tensor y = dev->new_tensor_by_vector({1}, y_data);
      const Tensor gy = dev->new_tensor_by_vector({1}, gy_data);
      Tensor gx = dev->new_tensor_by_constant({n}, 1);
      dev->min_bw(x, y, gy, 0, gx);
      EXPECT_TRUE(vector_match(expected, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckTranspose11) {
  const vector<float> gx_data {42};
  const vector<float> gy_data {42};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant({}, 0);
    const Tensor y = dev->transpose_fw(x);
    const Tensor gy = dev->new_tensor_by_vector({}, gy_data);
    Tensor gx = dev->new_tensor_by_constant({}, 0);
    dev->transpose_bw(x, y, gy, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckTransposeN1) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant({12}, 0);
    const Tensor y = dev->transpose_fw(x);
    const Tensor gy = dev->new_tensor_by_vector({1, 12}, gy_data);
    Tensor gx = dev->new_tensor_by_constant({12}, 0);
    dev->transpose_bw(x, y, gy, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckTranspose1N) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({1, 3}, 4), 0);
    const Tensor y = dev->transpose_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(Shape({3}, 4), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({1, 3}, 4), 0);
    dev->transpose_bw(x, y, gy, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckTransposeNN) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 3), 0);
    const Tensor y = dev->transpose_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(Shape({2, 2}, 3), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({2, 2}, 3), 0);
    dev->transpose_bw(x, y, gy, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckTransposeMN) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({2, 3}, 2), 0);
    const Tensor y = dev->transpose_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(Shape({3, 2}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({2, 3}, 2), 0);
    dev->transpose_bw(x, y, gy, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDims111) {
  const vector<float> gx_data {42, 43};
  const vector<float> gy_data {42, 43};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({}, 2), 0);
    const Tensor y = dev->permute_dims_fw(x, {0});
    const Tensor gy = dev->new_tensor_by_vector(Shape({}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({}, 2), 0);
    dev->permute_dims_bw(x, y, gy, {0}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDimsN11) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({6}, 2), 0);
    const Tensor y = dev->permute_dims_fw(x, {1, 0});
    const Tensor gy = dev->new_tensor_by_vector(Shape({1, 6}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({6}, 2), 0);
    dev->permute_dims_bw(x, y, gy, {1, 0}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDims1N1) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({1, 4}, 3), 0);
    const Tensor y = dev->permute_dims_fw(x, {0, 2, 1});
    const Tensor gy = dev->new_tensor_by_vector(Shape({1, 1, 4}, 3), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({1, 4}, 3), 0);
    dev->permute_dims_bw(x, y, gy, {0, 2, 1}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDims11N) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({1, 1, 4}, 3), 0);
    const Tensor y = dev->permute_dims_fw(x, {2, 0, 1});
    const Tensor gy = dev->new_tensor_by_vector(Shape({4}, 3), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({1, 1, 4}, 3), 0);
    dev->permute_dims_bw(x, y, gy, {2, 0, 1}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDimsMN1) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({2, 3}, 2), 0);
    const Tensor y = dev->permute_dims_fw(x, {1, 2, 0});
    const Tensor gy = dev->new_tensor_by_vector(Shape({3, 1, 2}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({2, 3}, 2), 0);
    dev->permute_dims_bw(x, y, gy, {1, 2, 0}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDimsM1N) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({3, 1, 2}, 2), 0);
    const Tensor y = dev->permute_dims_fw(x, {0, 2, 1});
    const Tensor gy = dev->new_tensor_by_vector(Shape({3, 2}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({3, 1, 2}, 2), 0);
    dev->permute_dims_bw(x, y, gy, {0, 2, 1}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDims1MN) {
  const vector<float> gx_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> gy_data {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({1, 3, 2}, 2), 0);
    const Tensor y = dev->permute_dims_fw(x, {2, 0, 1});
    const Tensor gy = dev->new_tensor_by_vector(Shape({2, 1, 3}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({1, 3, 2}, 2), 0);
    dev->permute_dims_bw(x, y, gy, {2, 0, 1}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckPermuteDimsLMN) {
  const vector<float> gx_data {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
  };
  const vector<float> gy_data {
    1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12,
    13, 19, 14, 20, 15, 21, 16, 22, 17, 23, 18, 24,
  };
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant(Shape({2, 3, 2}, 2), 0);
    const Tensor y = dev->permute_dims_fw(x, {2, 0, 1});
    const Tensor gy = dev->new_tensor_by_vector(Shape({2, 2, 3}, 2), gy_data);
    Tensor gx = dev->new_tensor_by_constant(Shape({2, 3, 2}, 2), 0);
    dev->permute_dims_bw(x, y, gy, {2, 0, 1}, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckFlip) {
  for (Device *dev : devices) {
    const vector<float> gy_data {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    };
    const vector<vector<float>> gx_data {
      {
        3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10,
        15, 14, 13, 18, 17, 16, 21, 20, 19, 24, 23, 22,
      },
      {
        4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9,
        16, 17, 18, 13, 14, 15, 22, 23, 24, 19, 20, 21,
      },
      {
        7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6,
        19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18,
      },
      {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      },
    };
    const Tensor gy = dev->new_tensor_by_vector(Shape({3, 2, 2}, 2), gy_data);
    for (std::uint32_t i = 0; i < 4; ++i) {
      Tensor gx = dev->new_tensor_by_constant(Shape({3, 2, 2}, 2), 1);
      dev->flip_bw(gy, i, gx);
      EXPECT_TRUE(vector_match(gx_data[i], gx.to_vector()));
    }
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
  }
}

TEST_F(TensorBackwardTest, CheckPowNN) {
  auto fga = [](float a, float b) { return b * std::pow(a, b - 1); };
  auto fgb = [](float a, float b) { return std::log(a) * std::pow(a, b); };

  const vector<float> a_val {1, 2, 4, 8, 16, 32, 64, 128};
  const vector<float> b_val {1, .5, .25, 0, -.125, -.25, -.5, -1.};
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 8192
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(ga_val, ga.to_vector(), ulps));
    EXPECT_TRUE(vector_match_ulps(gb_val, gb.to_vector(), ulps));
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

TEST_F(TensorBackwardTest, CheckBatchPickNN) {
  const vector<float> a_data {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    vector<std::uint32_t> ids;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {0, 0, 0},
      {6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10, 11}},
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {1, 1, 2},
      {0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14}},
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {0, 1, 0},
      {4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11}},
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {1, 0, 1},
      {2, 3, 4, 5, 8, 9, 10, 11, 8, 9, 10, 11}},
    {Shape({2, 2}, 2), {1, 1, 1, 1, 2, 2, 2, 2}, {2, 1},
      {0, 1, 2, 3, 6, 7, 8, 9, 9, 10, 11, 12}},
    {Shape({2, 2}, 4), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4},
      {1, 2, 0, 1}, {3, 4, 5, 6, 9, 10, 11, 12, 10, 11, 12, 13}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.b_shape, tc.b_data);
      dev->batch_pick_bw(b, tc.ids, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckBatchPickN1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    vector<std::uint32_t> ids;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 1), {1, 1, 2, 2}, {0},
      {1, 2, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3}},
    {Shape({2, 2}, 1), {1, 1, 2, 2}, {1},
      {0, 1, 2, 3, 1, 2, 4, 5, 0, 1, 2, 3}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.b_shape, tc.b_data);
      dev->batch_pick_bw(b, tc.ids, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckBatchPick1N) {
  const vector<float> a_data {0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    vector<std::uint32_t> ids;
    vector<float> y_data;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}, {0, 0, 0},
      {6, 7, 8, 9}},
    {Shape({2, 2}, 2), {1, 1, 1, 1, 2, 2, 2, 2}, {0, 0},
      {3, 4, 5, 6}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.b_shape, tc.b_data);
      dev->batch_pick_bw(b, tc.ids, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckInvalidBatchPick) {
  struct TestCase {
    Shape a_shape, b_shape;
    vector<std::uint32_t> ids;
  };
  vector<TestCase> test_cases {
    // Out-of-range IDs.
    {Shape({2, 2}, 3), {}, {}},
    {{}, {}, {1}},
    {{}, Shape({}, 3), {0, 0, 1}},
    {Shape({}, 3), {}, {3}},
    {Shape({}, 3), Shape({}, 3), {1, 0, 3}},
    // // Dims mismatched.
    {{}, {2}, {0}},
    {{}, Shape({2}, 3), {0, 0, 0}},
    {Shape({2}, 3), {}, {0}},
    {Shape({2}, 3), Shape({}, 3), {0, 0, 0}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_constant(tc.a_shape, 0);
      const Tensor b = dev->new_tensor_by_constant(tc.b_shape, 0);
      EXPECT_THROW(dev->batch_pick_bw(b, tc.ids, a), Error);
    }
  }
}

TEST_F(TensorBackwardTest, CheckCopyAndBatchPick) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {6, 7, 8, 9, 0, 1, 2, 3, 0, 1, 2, 3};
  for (Device *dev : devices) {
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

    const Tensor copied = a;

    dev->batch_pick_bw(b, {0, 0, 0}, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckBatchSliceNN_1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
    dev->batch_slice_bw(b, 0, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckBatchSliceNN_2) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 2, 2, 3, 3, 4, 4};
  struct TestCase {
    Shape shape;
    std::uint32_t offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {Shape({2, 2}, 2), 0, {1, 2, 4, 5, 3, 4, 6, 7, 0, 1, 2, 3}},
    {Shape({2, 2}, 2), 1, {0, 1, 2, 3, 1, 2, 4, 5, 3, 4, 6, 7}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.shape, b_data);
      dev->batch_slice_bw(b, tc.offset, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckBatchSliceN1_1) {
  const vector<float> a_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> b_data {-1, -2, -3, -4};
  const vector<float> y_data {0, 0, 0, 0, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
    dev->batch_slice_bw(b, 0, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckBatchSliceN1_2) {
  const vector<float> a_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> b_data {-1, -2, -3, -4};
  struct TestCase {
    Shape shape;
    std::uint32_t offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {{2, 2}, 0, {0, 0, 0, 0, 2, 3, 4, 5, 3, 4, 5, 6}},
    {{2, 2}, 1, {1, 2, 3, 4, 1, 1, 1, 1, 3, 4, 5, 6}},
    {{2, 2}, 2, {1, 2, 3, 4, 2, 3, 4, 5, 2, 2, 2, 2}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(tc.shape, b_data);
      dev->batch_slice_bw(b, tc.offset, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckInvalidBatchSlice) {
  struct TestCase {
    Shape a_shape, b_shape;
    std::uint32_t offset;
    bool ok;
  };
  vector<TestCase> test_cases {
    {Shape({}, 3), Shape({}, 2), 0, true},
    {Shape({}, 2), Shape({}, 3), 0, false},
    {Shape({42}, 3), Shape({42}, 2), 0, true},
    {Shape({42}, 3), Shape({41}, 2), 0, false},
    {{}, {}, 0, true},
    {{}, {}, 1, false},
    {{42}, {}, 0, false},
    {{42}, {}, 1, false},
    {{42}, {42}, 0, true},
    {{42}, {42}, 1, false},
    {{42}, {43}, 0, false},
    {{42}, {4, 42}, 0, false},
    {{42}, {4, 2, 42}, 0, false},
    {Shape({4, 4}, 3), Shape({4, 4}, 2), 0, true},
    {Shape({4, 4}, 3), Shape({4, 4}, 2), 1, true},
    {Shape({4, 4}, 3), Shape({4, 4}, 2), 2, false},
    {Shape({4, 4}, 3), Shape({4, 4}, 2), 3, false},
    {Shape({4, 4}, 3), Shape({4, 4}, 1), 0, true},
    {Shape({4, 4}, 3), Shape({4, 4}, 1), 1, true},
    {Shape({4, 4}, 3), Shape({4, 4}, 1), 2, true},
    {Shape({4, 4}, 3), Shape({4, 4}, 1), 3, false},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor_by_constant(tc.a_shape, 0);
      const Tensor b = dev->new_tensor_by_constant(tc.b_shape, 0);
      if (tc.ok) {
        EXPECT_NO_THROW(dev->batch_slice_bw(b, tc.offset, a));
      } else {
        EXPECT_THROW(dev->batch_slice_bw(b, tc.offset, a), Error);
      }
    }
  }
}

TEST_F(TensorBackwardTest, CheckCopyAndBatchSlice) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

    const Tensor copied = a;

    dev->batch_slice_bw(b, 0, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
  }
}

#define TEST_CONV2D(pad0, pad1, str0, str1, dil0, dil1) { \
  const vector<float> x_data = make_iota_vector(x_shape.size(), 1); \
  const vector<float> w_data = make_iota_vector(w_shape.size(), 1); \
  const vector<float> gy_data(y_shape.size(), 1); \
  for (Device *dev : devices) try { \
    const Tensor x = dev->new_tensor_by_vector(x_shape, x_data); \
    const Tensor w = dev->new_tensor_by_vector(w_shape, w_data); \
    const Tensor y = dev->conv2d_fw( \
        x, w, pad0, pad1, str0, str1, dil0, dil1); \
    const Tensor gy = dev->new_tensor_by_vector(y_shape, gy_data); \
    Tensor gx = dev->new_tensor_by_constant(x_shape, 1); \
    Tensor gw = dev->new_tensor_by_constant(w_shape, 1); \
    dev->conv2d_bw( \
        x, w, y, gy, pad0, pad1, str0, str1, dil0, dil1, gx, gw); \
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector())); \
    EXPECT_TRUE(vector_match(gw_data, gw.to_vector())); \
  } IGNORE_NOT_IMPLEMENTED \
}

TEST_F(TensorBackwardTest, CheckConv2D_1x1x1_1x1x1x1) {
  const Shape x_shape {};
  const Shape w_shape {};
  const Shape y_shape {};
  const vector<float> gx_data {2};
  const vector<float> gw_data {2};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x1x1_1x1x1x1) {
  const Shape x_shape {5};
  const Shape w_shape {};
  const Shape y_shape {5};
  const vector<float> gx_data {2, 2, 2, 2, 2};
  const vector<float> gw_data {16};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x1x1_2x1x1x1) {
  const Shape x_shape {5};
  const Shape w_shape {2};
  const Shape y_shape {4};
  const vector<float> gx_data {3, 4, 4, 4, 2};
  const vector<float> gw_data {15, 11};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x1x1_5x1x1x1) {
  const Shape x_shape {5};
  const Shape w_shape {5};
  const Shape y_shape {};
  const vector<float> gx_data {6, 5, 4, 3, 2};
  const vector<float> gw_data {6, 5, 4, 3, 2};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_1x5x1_1x1x1x1) {
  const Shape x_shape {1, 5};
  const Shape w_shape {};
  const Shape y_shape {1, 5};
  const vector<float> gx_data {2, 2, 2, 2, 2};
  const vector<float> gw_data {16};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_1x5x1_1x2x1x1) {
  const Shape x_shape {1, 5};
  const Shape w_shape {1, 2};
  const Shape y_shape {1, 4};
  const vector<float> gx_data {3, 4, 4, 4, 2};
  const vector<float> gw_data {15, 11};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_1x5x1_1x5x1x1) {
  const Shape x_shape {1, 5};
  const Shape w_shape {1, 5};
  const Shape y_shape {};
  const vector<float> gx_data {6, 5, 4, 3, 2};
  const vector<float> gw_data {6, 5, 4, 3, 2};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_1x1x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {};
  const Shape y_shape {5, 5};
  const vector<float> gx_data {
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
  };
  const vector<float> gw_data {326};
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x1x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2};
  const Shape y_shape {4, 5};
  const vector<float> gx_data {
    3, 4, 4, 4, 2,
    3, 4, 4, 4, 2,
    3, 4, 4, 4, 2,
    3, 4, 4, 4, 2,
    3, 4, 4, 4, 2,
  };
  const vector<float> gw_data {
    271, 251,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_5x1x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {5, 1};
  const Shape y_shape {1, 5};
  const vector<float> gx_data {
    6, 5, 4, 3, 2,
    6, 5, 4, 3, 2,
    6, 5, 4, 3, 2,
    6, 5, 4, 3, 2,
    6, 5, 4, 3, 2,
  };
  const vector<float> gw_data {
    76, 71, 66, 61, 56,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_1x2x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {1, 2};
  const Shape y_shape {5, 4};
  const vector<float> gx_data {
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    2, 2, 2, 2, 2,
  };
  const vector<float> gw_data {
    311,
    211,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {4, 4};
  const vector<float> gx_data {
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
  };
  const vector<float> gw_data {
    257, 241,
    177, 161,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_5x2x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {5, 2};
  const Shape y_shape {1, 4};
  const vector<float> gx_data {
    11, 10,  9,  8, 7,
    16, 14, 12, 10, 8,
    16, 14, 12, 10, 8,
    16, 14, 12, 10, 8,
     6,  5,  4,  3, 2,
  };
  const vector<float> gw_data {
    71, 67, 63, 59, 55,
    51, 47, 43, 39, 35,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_1x5x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {1, 5};
  const Shape y_shape {5};
  const vector<float> gx_data {
    6, 6, 6, 6, 6,
    5, 5, 5, 5, 5,
    4, 4, 4, 4, 4,
    3, 3, 3, 3, 3,
    2, 2, 2, 2, 2,
  };
  const vector<float> gw_data {
    116,
     91,
     66,
     41,
     16,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x5x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 5};
  const Shape y_shape {4};
  const vector<float> gx_data {
    11, 20, 20, 20, 10,
     9, 16, 16, 16,  8,
     7, 12, 12, 12,  6,
     5,  8,  8,  8,  4,
     3,  4,  4,  4,  2,
  };
  const vector<float> gw_data {
    95, 91,
    75, 71,
    55, 51,
    35, 31,
    15, 11,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_5x5x1x1) {
  const Shape x_shape {5, 5};
  const Shape w_shape {5, 5};
  const Shape y_shape {};
  const vector<float> gx_data {
    26, 25, 24, 23, 22,
    21, 20, 19, 18, 17,
    16, 15, 14, 13, 12,
    11, 10,  9,  8,  7,
     6,  5,  4,  3,  2,
  };
  const vector<float> gw_data {
    26, 25, 24, 23, 22,
    21, 20, 19, 18, 17,
    16, 15, 14, 13, 12,
    11, 10,  9,  8,  7,
     6,  5,  4,  3,  2,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x3_2x2x3x1) {
  const Shape x_shape {5, 5, 3};
  const Shape w_shape {2, 2, 3};
  const Shape y_shape {4, 4};
  const vector<float> gx_data {
    // channel 1
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
    // channel 2
     9, 16, 16, 16,  8,
    15, 27, 27, 27, 13,
    15, 27, 27, 27, 13,
    15, 27, 27, 27, 13,
     7, 12, 12, 12,  6,
    // channel 3
    13, 24, 24, 24, 12,
    23, 43, 43, 43, 21,
    23, 43, 43, 43, 21,
    23, 43, 43, 43, 21,
    11, 20, 20, 20, 10,
  };
  const vector<float> gw_data {
    // channel 1-1
    257, 241,
    177, 161,
    // channel 2-1
    657, 641,
    577, 561,
    // channel 3-1
    1057, 1041,
     977,  961,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x3) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2, 1, 3};
  const Shape y_shape {4, 4, 3};
  const vector<float> gx_data {
    25, 46, 46, 46, 22,
    43, 79, 79, 79, 37,
    43, 79, 79, 79, 37,
    43, 79, 79, 79, 37,
    19, 34, 34, 34, 16,
  };
  const vector<float> gw_data {
    // channel 1-1
    257, 241,
    177, 161,
    // channel 2-1
    257, 241,
    177, 161,
    // channel 3-1
    257, 241,
    177, 161,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x3_2x2x3x3) {
  const Shape x_shape {5, 5, 3};
  const Shape w_shape {2, 2, 3, 3};
  const Shape y_shape {4, 4, 3};
  const vector<float> gx_data {
    // channel 1
    49,  94,  94,  94, 46,
    91, 175, 175, 175, 85,
    91, 175, 175, 175, 85,
    91, 175, 175, 175, 85,
    43,  82,  82,  82, 40,
    // channel 2
     61, 118, 118, 118,  58,
    115, 223, 223, 223, 109,
    115, 223, 223, 223, 109,
    115, 223, 223, 223, 109,
     55, 106, 106, 106,  52,
    // channel 3
     73, 142, 142, 142,  70,
    139, 271, 271, 271, 133,
    139, 271, 271, 271, 133,
    139, 271, 271, 271, 133,
     67, 130, 130, 130,  64,
  };
  const vector<float> gw_data {
    // channel 1-1
    257, 241,
    177, 161,
    // channel 2-1
    657, 641,
    577, 561,
    // channel 3-1
    1057, 1041,
     977,  961,
    // channel 1-2
    257, 241,
    177, 161,
    // channel 2-2
    657, 641,
    577, 561,
    // channel 3-2
    1057, 1041,
     977,  961,
    // channel 1-3
    257, 241,
    177, 161,
    // channel 2-3
    657, 641,
    577, 561,
    // channel 3-3
    1057, 1041,
     977,  961,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Padding10) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {6, 4};
  const vector<float> gx_data {
     8,  8,  8,  8,  8,
    11, 11, 11, 11, 11,
    11, 11, 11, 11, 11,
    11, 11, 11, 11, 11,
     4,  4,  4,  4,  4,
  };
  const vector<float> gw_data {
    311, 311,
    211, 211,
  };
  TEST_CONV2D(1, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Padding01) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {4, 6};
  const vector<float> gx_data {
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
  };
  const vector<float> gw_data {
    271, 251,
    271, 251,
  };
  TEST_CONV2D(0, 1, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Padding11) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {6, 6};
  const vector<float> gx_data {
    11, 11, 11, 11, 11,
    11, 11, 11, 11, 11,
    11, 11, 11, 11, 11,
    11, 11, 11, 11, 11,
    11, 11, 11, 11, 11,
  };
  const vector<float> gw_data {
    326, 326,
    326, 326,
  };
  TEST_CONV2D(1, 1, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Stride21) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {2, 4};
  const vector<float> gx_data {
    5, 4, 5, 4, 1,
    7, 5, 7, 5, 1,
    7, 5, 7, 5, 1,
    7, 5, 7, 5, 1,
    3, 2, 3, 2, 1,
  };
  const vector<float> gw_data {
    125, 117,
     85,  77,
  };
  TEST_CONV2D(0, 0, 2, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Stride12) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {4, 2};
  const vector<float> gx_data {
    5, 8, 8, 8, 4,
    3, 4, 4, 4, 2,
    5, 8, 8, 8, 4,
    3, 4, 4, 4, 2,
    1, 1, 1, 1, 1,
  };
  const vector<float> gw_data {
    109, 101,
     69,  61,
  };
  TEST_CONV2D(0, 0, 1, 2, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Stride22) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {2, 2};
  const vector<float> gx_data {
    5, 4, 5, 4, 1,
    3, 2, 3, 2, 1,
    5, 4, 5, 4, 1,
    3, 2, 3, 2, 1,
    1, 1, 1, 1, 1,
  };
  const vector<float> gw_data {
    53, 49,
    33, 29,
  };
  TEST_CONV2D(0, 0, 2, 2, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Dilation21) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {3, 4};
  const vector<float> gx_data {
    5, 5,  8, 4, 4,
    7, 7, 11, 5, 5,
    7, 7, 11, 5, 5,
    7, 7, 11, 5, 5,
    3, 3,  4, 2, 2,
  };
  const vector<float> gw_data {
    199, 175,
    139, 115,
  };
  TEST_CONV2D(0, 0, 1, 1, 2, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Dilation12) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {4, 3};
  const vector<float> gx_data {
    5,  8,  8,  8, 4,
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
    3,  4,  4,  4, 2,
  };
  const vector<float> gw_data {
    223, 211,
    103,  91,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 2);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_Dilation22) {
  const Shape x_shape {5, 5};
  const Shape w_shape {2, 2};
  const Shape y_shape {3, 3};
  const vector<float> gx_data {
    5, 5,  8, 4, 4,
    5, 5,  8, 4, 4,
    7, 7, 11, 5, 5,
    3, 3,  4, 2, 2,
    3, 3,  4, 2, 2,
  };
  const vector<float> gw_data {
    172, 154,
     82,  64,
  };
  TEST_CONV2D(0, 0, 1, 1, 2, 2);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_N1) {
  const Shape x_shape({5, 5}, 3);
  const Shape w_shape {2, 2};
  const Shape y_shape({4, 4}, 3);
  const vector<float> gx_data {
    // minibatch 1
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
    // minibatch 2
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
    // minibatch 3
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
  };
  const vector<float> gw_data {
    1969, 1921,
    1729, 1681,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_1N) {
  const Shape x_shape {5, 5};
  const Shape w_shape({2, 2}, 3);
  const Shape y_shape({4, 4}, 3);
  const vector<float> gx_data {
    25, 46, 46, 46, 22,
    43, 79, 79, 79, 37,
    43, 79, 79, 79, 37,
    43, 79, 79, 79, 37,
    19, 34, 34, 34, 16,
  };
  const vector<float> gw_data {
    // minibatch 1
    257, 241,
    177, 161,
    // minibatch 2
    257, 241,
    177, 161,
    // minibatch 3
    257, 241,
    177, 161,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckConv2D_5x5x1_2x2x1x1_NN) {
  const Shape x_shape({5, 5}, 3);
  const Shape w_shape({2, 2}, 3);
  const Shape y_shape({4, 4}, 3);
  const vector<float> gx_data {
    // minibatch 1
    5,  8,  8,  8, 4,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    7, 11, 11, 11, 5,
    3,  4,  4,  4, 2,
    // minibatch 2
     9, 16, 16, 16,  8,
    15, 27, 27, 27, 13,
    15, 27, 27, 27, 13,
    15, 27, 27, 27, 13,
     7, 12, 12, 12,  6,
    // minibatch 3
    13, 24, 24, 24, 12,
    23, 43, 43, 43, 21,
    23, 43, 43, 43, 21,
    23, 43, 43, 43, 21,
    11, 20, 20, 20, 10,
  };
  const vector<float> gw_data {
    // minibatch 1
    257, 241,
    177, 161,
    // minibatch 2
    657, 641,
    577, 561,
    // minibatch 3
    1057, 1041,
     977,  961,
  };
  TEST_CONV2D(0, 0, 1, 1, 1, 1);
}

#undef TEST_CONV2D

TEST_F(TensorBackwardTest, CheckConv2D_VGG16FirstLayer) {
  const Shape x_shape {224, 224, 3};
  const Shape w_shape {3, 3, 3, 64};
  const Shape y_shape {224, 224, 64};
  vector<float> gx_data(224 * 224 * 3, 1 + 3 * 3 * 64);
  for (unsigned b = 0; b < 3; ++b) {
    float *pgx = gx_data.data() + b * 224 * 224;
    pgx[0] += 64;
    pgx[223] += 64;
    pgx[223 * 224] += 64;
    pgx[223 * 224 + 223] += 64;
    for (unsigned i = 0; i < 224; ++i) {
      pgx[i] -= 3 * 64;
      pgx[223 * 224 + i] -= 3 * 64;
      pgx[i * 224] -= 3 * 64;
      pgx[i * 224 + 223] -= 3 * 64;
    }
  }
  vector<float> gw_data(3 * 3 * 3 * 64, 1 + 224 * 224);
  for (unsigned b = 0; b < 3 * 64; ++b) {
    float *pgw = gw_data.data() + b * 3 * 3;
    pgw[0] += -2 * 224 + 1;
    pgw[1] += -224;
    pgw[2] += -2 * 224 + 1;
    pgw[3] += -224;
    //pgw[4] += 0;
    pgw[5] += -224;
    pgw[6] += -2 * 224 + 1;
    pgw[7] += -224;
    pgw[8] += -2 * 224 + 1;
  }

  const vector<float> x_data(x_shape.size(), 1);
  const vector<float> w_data(w_shape.size(), 1);
  const vector<float> gy_data(y_shape.size(), 1);

  for (Device *dev : devices) try {
    const Tensor x = dev->new_tensor_by_vector(x_shape, x_data);
    const Tensor w = dev->new_tensor_by_vector(w_shape, w_data);
    const Tensor y = dev->conv2d_fw(x, w, 1, 1, 1, 1, 1, 1);
    const Tensor gy = dev->new_tensor_by_vector(y_shape, gy_data);
    Tensor gx = dev->new_tensor_by_constant(x_shape, 1);
    Tensor gw = dev->new_tensor_by_constant(w_shape, 1);
    dev->conv2d_bw(x, w, y, gy, 1, 1, 1, 1, 1, 1, gx, gw);

    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));

    const auto dev_type = dev->type();
    const std::uint32_t ulps
      = dev_type == DeviceType::CUDA16 ? 65536
      : get_default_ulps(*dev);
    EXPECT_TRUE(vector_match_ulps(gw_data, gw.to_vector(), ulps));
  } IGNORE_NOT_IMPLEMENTED
}

#define TEST_MAX_POOL2D(win0, win1, pad0, pad1, str0, str1) { \
  const vector<float> x_data = make_iota_vector(x_shape.size(), 1); \
  const vector<float> gy_data(y_shape.size(), 1); \
  for (Device *dev : devices) try { \
    const Tensor x = dev->new_tensor_by_vector(x_shape, x_data); \
    const Tensor y = dev->max_pool2d_fw( \
        x, win0, win1, pad0, pad1, str0, str1); \
    const Tensor gy = dev->new_tensor_by_vector(y_shape, gy_data); \
    Tensor gx = dev->new_tensor_by_constant(x_shape, 1); \
    dev->max_pool2d_bw(x, y, gy, win0, win1, pad0, pad1, str0, str1, gx); \
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector())); \
  } IGNORE_NOT_IMPLEMENTED \
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_1x1x1_1x1) {
  const Shape x_shape {};
  const Shape y_shape {};
  const vector<float> gx_data {2};
  TEST_MAX_POOL2D(1, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x1x1_1x1) {
  const Shape x_shape {5};
  const Shape y_shape {5};
  const vector<float> gx_data {2, 2, 2, 2, 2};
  TEST_MAX_POOL2D(1, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x1x1_2x1) {
  const Shape x_shape {5};
  const Shape y_shape {4};
  const vector<float> gx_data {1, 2, 2, 2, 2};
  TEST_MAX_POOL2D(2, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x1x1_5x1) {
  const Shape x_shape {5};
  const Shape y_shape {};
  const vector<float> gx_data {1, 1, 1, 1, 2};
  TEST_MAX_POOL2D(5, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_1x5x1_1x1) {
  const Shape x_shape {1, 5};
  const Shape y_shape {1, 5};
  const vector<float> gx_data {2, 2, 2, 2, 2};
  TEST_MAX_POOL2D(1, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_1x5x1_1x2) {
  const Shape x_shape {1, 5};
  const Shape y_shape {1, 4};
  const vector<float> gx_data {1, 2, 2, 2, 2};
  TEST_MAX_POOL2D(1, 2, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_1x5x1_1x5) {
  const Shape x_shape {1, 5};
  const Shape y_shape {};
  const vector<float> gx_data {1, 1, 1, 1, 2};
  TEST_MAX_POOL2D(1, 5, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_1x1) {
  const Shape x_shape {5, 5};
  const Shape y_shape {5, 5};
  const vector<float> gx_data {
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(1, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x1) {
  const Shape x_shape {5, 5};
  const Shape y_shape {4, 5};
  const vector<float> gx_data {
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(2, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_5x1) {
  const Shape x_shape {5, 5};
  const Shape y_shape {1, 5};
  const vector<float> gx_data {
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
  };
  TEST_MAX_POOL2D(5, 1, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_1x2) {
  const Shape x_shape {5, 5};
  const Shape y_shape {5, 4};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
    2, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(1, 2, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2) {
  const Shape x_shape {5, 5};
  const Shape y_shape {4, 4};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(2, 2, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_5x2) {
  const Shape x_shape {5, 5};
  const Shape y_shape {1, 4};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
    1, 1, 1, 1, 2,
  };
  TEST_MAX_POOL2D(5, 2, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_1x5) {
  const Shape x_shape {5, 5};
  const Shape y_shape {5};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(1, 5, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x5) {
  const Shape x_shape {5, 5};
  const Shape y_shape {4};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(2, 5, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_5x5) {
  const Shape x_shape {5, 5};
  const Shape y_shape {};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 2,
  };
  TEST_MAX_POOL2D(5, 5, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x3_2x2) {
  const Shape x_shape {5, 5, 3};
  const Shape y_shape {4, 4, 3};
  const vector<float> gx_data {
    // channel 1
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    // channel 2
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    // channel 3
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(2, 2, 0, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_Padding10) {
  const Shape x_shape {5, 5};
  const Shape y_shape {6, 4};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 3,
    2, 2, 2, 2, 3,
    2, 2, 2, 2, 3,
    2, 2, 2, 2, 3,
  };
  TEST_MAX_POOL2D(2, 2, 1, 0, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_Padding01) {
  const Shape x_shape {5, 5};
  const Shape y_shape {4, 6};
  const vector<float> gx_data {
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 3, 3, 3, 3,
  };
  TEST_MAX_POOL2D(2, 2, 0, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_Padding11) {
  const Shape x_shape {5, 5};
  const Shape y_shape {6, 6};
  const vector<float> gx_data {
    2, 2, 2, 2, 3,
    2, 2, 2, 2, 3,
    2, 2, 2, 2, 3,
    2, 2, 2, 2, 3,
    3, 3, 3, 3, 5,
  };
  TEST_MAX_POOL2D(2, 2, 1, 1, 1, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_Stride21) {
  const Shape x_shape {5, 5};
  const Shape y_shape {2, 4};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 2, 1, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 1, 2, 1,
  };
  TEST_MAX_POOL2D(2, 2, 0, 0, 2, 1);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_Stride12) {
  const Shape x_shape {5, 5};
  const Shape y_shape {4, 2};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 1, 1, 1, 1,
  };
  TEST_MAX_POOL2D(2, 2, 0, 0, 1, 2);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_Stride22) {
  const Shape x_shape {5, 5};
  const Shape y_shape {2, 2};
  const vector<float> gx_data {
    1, 1, 1, 1, 1,
    1, 2, 1, 2, 1,
    1, 1, 1, 1, 1,
    1, 2, 1, 2, 1,
    1, 1, 1, 1, 1,
  };
  TEST_MAX_POOL2D(2, 2, 0, 0, 2, 2);
}

TEST_F(TensorBackwardTest, CheckMaxPool2D_5x5x1_2x2_N) {
  const Shape x_shape({5, 5}, 3);
  const Shape y_shape({4, 4}, 3);
  const vector<float> gx_data {
    // minibatch 1
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    // minibatch 2
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    // minibatch 3
    1, 1, 1, 1, 1,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
    1, 2, 2, 2, 2,
  };
  TEST_MAX_POOL2D(2, 2, 0, 0, 1, 1);
}

#undef TEST_MAX_POOL2D

TEST_F(TensorBackwardTest, CheckMaxPool2D_VGG16ThirdLayer) {
  const Shape x_shape {224, 224, 64};
  const Shape y_shape {112, 112, 64};

  vector<float> x_data(224 * 224 * 64);
  for (unsigned b = 0; b < 64; ++b) {
    float *px = x_data.data() + b * 224 * 224;
    for (unsigned x = 0; x < 224; ++x) {
      float *px2 = px + x * 224;
      for (unsigned y = 0; y < 224; ++y) {
        px2[y] = x + y;
      }
    }
  }

  vector<float> gx_data(224 * 224 * 64, 1);
  for (unsigned b = 0; b < 64; ++b) {
    float *pgx = gx_data.data() + b * 224 * 224;
    for (unsigned x = 1; x < 224; x += 2) {
      float *pgx2 = pgx + x * 224;
      for (unsigned y = 1; y < 224; y += 2) {
        pgx2[y] += 1;
      }
    }
  }

  const vector<float> gy_data(y_shape.size(), 1);

  for (Device *dev : devices) try {
    const Tensor x = dev->new_tensor_by_vector(x_shape, x_data);
    const Tensor y = dev->max_pool2d_fw(x, 2, 2, 0, 0, 2, 2);
    const Tensor gy = dev->new_tensor_by_vector(y_shape, gy_data);
    Tensor gx = dev->new_tensor_by_constant(x_shape, 1);
    dev->max_pool2d_bw(x, y, gy, 2, 2, 0, 0, 2, 2, gx);
    EXPECT_TRUE(vector_match(gx_data, gx.to_vector()));
  } IGNORE_NOT_IMPLEMENTED
}

}  // namespace primitiv
