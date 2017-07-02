#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

#ifdef USE_CUDA
#include <primitiv/cuda_device.h>
#endif  // USE_CUDA

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class TensorBackwardTest : public testing::Test {
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

TEST_F(TensorBackwardTest, CheckSliceNN_1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    for (unsigned i : {0, 1, 2, 5, 10}) {
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
    unsigned dim, offset;
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
    for (unsigned i : {0, 1, 2, 5, 10}) {
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
    unsigned dim, offset;
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
    for (unsigned i : {0, 1, 2, 5, 10}) {
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
    unsigned dim, offset;
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
    unsigned dim, offset;
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
      Tensor a = dev->new_tensor(tc.a_shape, 0);
      const Tensor b = dev->new_tensor(tc.b_shape, 0);
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
    for (unsigned i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

      const Tensor copied = a;
      EXPECT_EQ(static_cast<const Tensor>(a).data(), copied.data());

      dev->slice_bw(b, i, 0, a);
      EXPECT_NE(static_cast<const Tensor>(a).data(), copied.data());
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
    unsigned dim;
    vector<unsigned> ids;
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
      dev->pick_bw(b, tc.dim, tc.ids, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPickN1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    unsigned dim;
    vector<unsigned> ids;
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
      dev->pick_bw(b, tc.dim, tc.ids, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPick1N) {
  const vector<float> a_data {0, 1, 2, 3};
  struct TestCase {
    Shape b_shape;
    vector<float> b_data;
    unsigned dim;
    vector<unsigned> ids;
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
      dev->pick_bw(b, tc.dim, tc.ids, a);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckInvalidPick) {
  struct TestCase {
    Shape a_shape, b_shape;
    unsigned dim;
    vector<unsigned> ids;
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
      Tensor a = dev->new_tensor(tc.a_shape, 0);
      const Tensor b = dev->new_tensor(tc.b_shape, 0);
      EXPECT_THROW(dev->pick_bw(b, tc.dim, tc.ids, a), Error);
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
    EXPECT_EQ(static_cast<const Tensor>(a).data(), copied.data());

    dev->pick_bw(b, 2, {0, 0, 0}, a);
    EXPECT_NE(static_cast<const Tensor>(a).data(), copied.data());
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckNegate) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->negate_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor(x.shape(), 0);
    dev->negate_bw(x, y, gy, gx);
    const vector<float> gx_val {-1, 1, -2, 2, -2, 2, -1, 1};
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSqrt) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0.01, 1, 4, 9, 0.01, 1, 4, 9});
    const Tensor y = dev->sqrt_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor(x.shape(), 0);
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
    Tensor gx = dev->new_tensor(x.shape(), 0);
    dev->exp_bw(x, y, gy, gx);
    const vector<float> gx_val {
      1, -2.7182818, 14.778112, -40.171074,
      2, -.73575888, .13533528, -.049787068,
    };
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
    Tensor gx = dev->new_tensor(x.shape(), 0);
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
    Tensor gx = dev->new_tensor(x.shape(), 0);
    dev->sigmoid_bw(x, y, gy, gx);
    const vector<float> gx_val {
      .25, -.19661193, .20998717, -.090353319,
      .5, -.39322387, .10499359, -.045176660,
    };
    EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckSin) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
    const Tensor y = dev->sin_fw(x);
    const Tensor gy = dev->new_tensor_by_vector(
        y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor gx = dev->new_tensor(x.shape(), 0);
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
    Tensor gx = dev->new_tensor(x.shape(), 0);
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
    Tensor gx = dev->new_tensor(x.shape(), 0);
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
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->add_const_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
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
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->subtract_const_r_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
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
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->subtract_const_l_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
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
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->multiply_const_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
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
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->divide_const_r_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
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
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {.1, 1, 2, 3, -.1, -1, -2, -3});
      const Tensor y = dev->divide_const_l_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
      dev->divide_const_l_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
       -100 * k, k, -k / 2, 2 * k / 9, -200 * k, 2 * k, -k / 4, k / 9,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckPReLU) {
  const vector<float> ks {.01, .1, 1., 10., 100., -.01, -.1, -1., -10., -100.};
  for (Device *dev :devices) {
    for (const float k : ks) {
      const Tensor x = dev->new_tensor_by_vector(
          Shape({2, 2}, 2), {0, 1, 2, 3, 0, -1, -2, -3});
      const Tensor y = dev->prelu_fw(x, k);
      const Tensor gy = dev->new_tensor_by_vector(
          y.shape(), {1, -1, 2, -2, 2, -2, 1, -1});
      Tensor gx = dev->new_tensor(x.shape(), 0);
      dev->prelu_bw(x, y, gy, k, gx);
      const vector<float> gx_val {
        k, -1, 2, -2, 2 * k, -2 * k, k, -k,
      };
      EXPECT_TRUE(vector_match(gx_val, gx.to_vector()));
    }
  }
}

TEST_F(TensorBackwardTest, CheckMatMul11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, {1, 2, 3, 4});
    const Tensor b = dev->new_tensor_by_vector({2, 2}, {1, 0, 0, 2});
    const Tensor gy = dev->new_tensor_by_vector({2, 2}, {1, -1, 2, -2});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
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
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
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
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
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
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4, 2, -2, 2, -2};
    const vector<float> gb_val {1, 1, -1, -1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

}  // namespace primitiv
