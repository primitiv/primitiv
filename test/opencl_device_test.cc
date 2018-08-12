#include <primitiv/config.h>

#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/devices/opencl/device.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class OpenCLDeviceTest : public testing::Test {
protected:
  struct Config {
    std::uint32_t pf_id;
    std::uint32_t dev_id;
  };
  vector<Config> configs;

  void SetUp() override {
    const std::uint32_t num_pfs = devices::OpenCL::num_platforms();
    for (std::uint32_t pf_id = 0; pf_id < num_pfs; ++pf_id) {
      const std::uint32_t num_devs = devices::OpenCL::num_devices(pf_id);
      for (std::uint32_t dev_id = 0; dev_id < num_devs; ++dev_id) {
        if (devices::OpenCL::check_support(pf_id, dev_id)) {
          configs.emplace_back(Config { pf_id, dev_id });
        }
      }
    }
    if (configs.empty()) {
      std::cerr << "No available OpenCL devices." << std::endl;
    }
  }
};

TEST_F(OpenCLDeviceTest, CheckDeviceType) {
  for (const Config &cfg : configs) {
    devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
    EXPECT_EQ(DeviceType::OPENCL, dev.type());
  }
}

TEST_F(OpenCLDeviceTest, CheckInvalidInit) {
  const std::uint32_t num_pfs = devices::OpenCL::num_platforms();
  EXPECT_THROW(devices::OpenCL(num_pfs, 0), Error);
  EXPECT_THROW(devices::OpenCL(12345678, 0), Error);
  for (std::uint32_t pf_id = 0; pf_id < num_pfs; ++pf_id) {
    EXPECT_THROW(
        devices::OpenCL(pf_id, devices::OpenCL::num_devices(pf_id)), Error);
    EXPECT_THROW(devices::OpenCL(pf_id, 12345678), Error);
  }
}

TEST_F(OpenCLDeviceTest, CheckNewDelete) {
  for (const Config &cfg : configs) {
    devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
    {
      // 1 value
      Tensor x1 = dev.new_tensor_by_constant(Shape(), 0);
      // 256 values
      Tensor x2 = dev.new_tensor_by_constant(Shape({16, 16}), 0);
      // 65536 values
      Tensor x3 = dev.new_tensor_by_constant(Shape({16, 16, 16}, 16), 0);
    }
    // All tensors are already deleted before arriving here.
  }
  SUCCEED();
}

TEST_F(OpenCLDeviceTest, CheckDanglingTensor) {
  for (const Config &cfg : configs) {
    Tensor x1;
    {
      devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
      x1 = dev.new_tensor_by_constant(Shape(), 0);
    }
    // x1 still has valid object,
    // but there is no guarantee that the memory is alive.
    // Our implementation only guarantees the safety to delete Tensors anytime.
  }
  SUCCEED();
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(OpenCLDeviceTest, CheckRandomBernoulli) {
  vector<vector<float>> history;
  for (const Config &cfg : configs) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
      const Tensor x = dev.random_bernoulli(Shape({3, 3}, 3), 0.3);
      const vector<float> x_val = x.to_vector();

      std::cout << "Epoch " << i << ':';
      for (float x_i : x_val) {
        std::cout << ' ' << x_i;
      }
      std::cout << std::endl;

      for (const vector<float> &h_val : history) {
        EXPECT_FALSE(vector_match(x_val, h_val));
      }
      history.emplace_back(x_val);

      // Wait for updating the device randomizer.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(OpenCLDeviceTest, CheckRandomBernoulliWithSeed) {
  const vector<float> expected {
    0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
  };
  for (const Config &cfg : configs) {
    devices::OpenCL dev(cfg.pf_id, cfg.dev_id, 12345);
    const Tensor x = dev.random_bernoulli(Shape({4, 4}, 4), 0.3);
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
  }
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(OpenCLDeviceTest, CheckRandomUniform) {
  vector<vector<float>> history;
  for (const Config &cfg : configs) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
      const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
      const vector<float> x_val = x.to_vector();

      std::cout << "Epoch " << i << ':';
      for (float x_i : x_val) {
        std::cout << ' ' << x_i;
      }
      std::cout << std::endl;

      for (const vector<float> &h_val : history) {
        EXPECT_FALSE(vector_match(x_val, h_val));
      }
      history.emplace_back(x_val);

      // Wait for updating the device randomizer.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(OpenCLDeviceTest, CheckRandomUniformWithSeed) {
  const vector<float> expected {
    7.7330894e+00, 7.0227852e+00, -3.3052402e+00, -6.6472688e+00,
    -5.6894612e+00, -8.2843294e+00, -5.3179150e+00, 5.8758497e+00,
  };
  for (const Config &cfg : configs) {
    devices::OpenCL dev(cfg.pf_id, cfg.dev_id, 12345);
    const Tensor x = dev.random_uniform(Shape({2, 2}, 2), -9, 9);
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
  }
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(OpenCLDeviceTest, CheckRandomNormal) {
  vector<vector<float>> history;
  for (const Config &cfg : configs) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
      const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
      const vector<float> x_val = x.to_vector();

      std::cout << "Epoch " << i << ':';
      for (float x_i : x_val) {
        std::cout << ' ' << x_i;
      }
      std::cout << std::endl;

      for (const vector<float> &h_val : history) {
        EXPECT_FALSE(vector_match(x_val, h_val));
      }
      history.emplace_back(x_val);

      // Wait for updating the device randomizer.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(OpenCLDeviceTest, CheckRandomNormalWithSeed) {
#ifdef __GLIBCXX__
  const vector<float> expected {
    -1.3574908e+00, -1.7222166e-01, 2.5865970e+00, -4.3594337e-01,
    4.5383353e+00, 8.4703674e+00, 2.5535507e+00, 1.3252910e+00,
  };
#elif defined _LIBCPP_VERSION
  const vector<float> expected {
    -1.7222166e-01, -1.3574908e+00, -4.3594337e-01, 2.5865970e+00,
    8.4703674e+00, 4.5383353e+00, 1.3252910e+00, 2.5535507e+00,
  };
#else
  const vector<float> expected {};
  std::cerr << "Unknown C++ library. Expected results can't be defined." << std::endl;
#endif
  for (const Config &cfg : configs) {
    devices::OpenCL dev(cfg.pf_id, cfg.dev_id, 12345);
    const Tensor x = dev.random_normal(Shape({2, 2}, 2), 1, 3);
#ifdef PRIMITIV_MAYBE_FPMATH_X87
    EXPECT_TRUE(vector_near(expected, x.to_vector(), 1e-6));
#else
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
#endif
  }
}

#ifdef PRIMITIV_BUILD_TESTS_PROBABILISTIC
TEST_F(OpenCLDeviceTest, CheckRandomLogNormal) {
  vector<vector<float>> history;
  for (const Config &cfg : configs) {
    for (std::uint32_t i = 0; i < 10; ++i) {
      devices::OpenCL dev(cfg.pf_id, cfg.dev_id);
      const Tensor x = dev.random_log_normal(Shape({2, 2}, 2), 1, 3);
      const vector<float> x_val = x.to_vector();

      std::cout << "Epoch " << i << ':';
      for (float x_i : x_val) {
        std::cout << ' ' << x_i;
      }
      std::cout << std::endl;

      for (const vector<float> &h_val : history) {
        EXPECT_FALSE(vector_match(x_val, h_val));
      }
      history.emplace_back(x_val);

      // Wait for updating the device randomizer.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
}
#endif  // PRIMITIV_BUILD_TESTS_PROBABILISTIC

TEST_F(OpenCLDeviceTest, CheckRandomLogNormalWithSeed) {
#ifdef __GLIBCXX__
  const vector<float> expected {
    2.5730559e-01, 8.4179258e-01, 1.3284487e+01, 6.4665437e-01,
    9.3534966e+01, 4.7712681e+03, 1.2852659e+01, 3.7632804e+00,
  };
#elif defined _LIBCPP_VERSION
  const vector<float> expected {
    8.4179258e-01, 2.5730559e-01, 6.4665437e-01, 1.3284487e+01,
    4.7712681e+03, 9.3534966e+01, 3.7632804e+00, 1.2852659e+01,
  };
#else
  const vector<float> expected {};
  std::cerr << "Unknown C++ library. Expected results can't be defined." << std::endl;
#endif
  for (const Config &cfg : configs) {
    devices::OpenCL dev(cfg.pf_id, cfg.dev_id, 12345);
    const Tensor x = dev.random_log_normal(Shape({2, 2}, 2), 1, 3);
#ifdef PRIMITIV_MAYBE_FPMATH_X87
    EXPECT_TRUE(vector_near(expected, x.to_vector(), 1e-4));
#else
    EXPECT_TRUE(vector_match(expected, x.to_vector()));
#endif
  }
}

}  // namespace primitiv
