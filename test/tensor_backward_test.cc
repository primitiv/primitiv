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
