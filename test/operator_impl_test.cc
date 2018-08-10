#include <primitiv/config.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/functions.h>
#include <primitiv/core/initializer_impl.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/operator_impl.h>
#include <primitiv/core/parameter.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {
namespace operators {

class OperatorImplTest : public testing::Test {
public:
  void setup_1arg() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
  }

  void setup_1arg_nonnegative() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, .01, .01, .01, .01, 1, 4, 9, 16})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
  }

  void setup_1arg_nonzero() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, 1, -1, 1, -1, -1, -2, -3, -4})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
  }

  void setup_2args() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[1], {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[1], *dev)));
  }

  void setup_2args_swapped() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[1], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[1], *dev)));
  }

  void setup_2args_scalar() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[1], {1, 2, 3})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[1], *dev)));
  }

  void setup_2args_scalar_nonzero() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, 1, -1, 1, -1, -1, -2, -3, -4})));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[1], {1, 2, 3})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[1], *dev)));
  }

  void setup_2args_scalar_nonnegative() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, .01, .01, .01, .01, 1, 4, 9, 16})));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[1], {1, 2, 3})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[1], *dev)));
  }

  void setup_2args_softmax_cross_entropy() {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
        *arg_shapes[1], {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1})));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
    arg_grads.emplace_back(
        new Tensor(functions::zeros<Tensor>(*arg_shapes[1], *dev)));
  }

  void reset_gradients() {
    for (Tensor *x : arg_grads) x->reset(0);
  }

protected:
  void SetUp() override {
    dev = new devices::Naive(12345);
  }

  void TearDown() override {
    for (const Shape *x : arg_shapes) delete x;
    for (const Tensor *x : arg_values) delete x;
    for (Tensor *x : arg_grads) delete x;
    delete dev;
  }

  Device *dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

#define COMMON_PROC \
  Shape cur_shape; \
  Tensor cur_value; \
  node.forward_shape(arg_shapes, { &cur_shape }); \
  node.forward(arg_values, { &cur_value }); \
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev); \
  node.backward(arg_values, { &cur_value } , { &cur_grad }, arg_grads);

#define COMMON_CHECK \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_EQ(nullptr, node.get_device()); \
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector())); \
  EXPECT_TRUE(vector_match(bw_grad, arg_grads[0]->to_vector()));

#define COMMON_CHECK_2ARGS \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_EQ(nullptr, node.get_device()); \
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector())); \
  EXPECT_TRUE(vector_match(bw_grads[0], arg_grads[0]->to_vector())); \
  EXPECT_TRUE(vector_match(bw_grads[1], arg_grads[1]->to_vector()));

#define COMMON_CHECK_NEAR(err) \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_EQ(nullptr, node.get_device()); \
  EXPECT_TRUE(vector_near(ret_data, cur_value.to_vector(), err)); \
  EXPECT_TRUE(vector_near(bw_grad, arg_grads[0]->to_vector(), err));

#define TEST_1ARG(name_) { \
  name_ node; \
  EXPECT_EQ(#name_, node.name()); \
  COMMON_PROC; \
  COMMON_CHECK; \
}

#define TEST_1ARG_K(name_, k_) { \
  name_ node(k_); \
  EXPECT_EQ( \
      std::string(#name_) + '(' + \
      std::to_string(static_cast<float>(k_)) + ')', node.name()); \
  COMMON_PROC; \
  COMMON_CHECK; \
}

#define TEST_1ARG_NEAR(name_, err) { \
  name_ node; \
  EXPECT_EQ(#name_, node.name()); \
  COMMON_PROC; \
  COMMON_CHECK_NEAR(err); \
}

#define TEST_1ARG_K_NEAR(name_, k_, err) { \
  name_ node(k_); \
  EXPECT_EQ( \
      std::string(#name_) + '(' + \
      std::to_string(static_cast<float>(k_)) + ')', node.name()); \
  COMMON_PROC; \
  COMMON_CHECK_NEAR(err); \
}

#define TEST_2ARGS(name_) { \
  name_ node; \
  EXPECT_EQ(#name_, node.name()); \
  COMMON_PROC; \
  COMMON_CHECK_2ARGS; \
}

TEST_F(OperatorImplTest, CheckInput) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  Input node(ret_shape, ret_data, *dev);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  // backward() has no effect.
  EXPECT_NO_THROW(node.backward(
        { &cur_value } , { &cur_grad }, arg_values, arg_grads));
  EXPECT_EQ("Input", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(dev, node.get_device());
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
}

TEST_F(OperatorImplTest, CheckParameter) {
  const Shape ret_shape {2, 2};
  const initializers::Constant init(42);
  primitiv::Parameter param(ret_shape, init, *dev);
  ASSERT_TRUE(vector_match(vector<float>(4, 42), param.value().to_vector()));
  ASSERT_TRUE(vector_match(vector<float>(4, 0), param.gradient().to_vector()));

  Parameter node(param);
  Shape cur_shape;
  node.forward_shape(arg_shapes, { &cur_shape });
  // Parameter could not return values from forward().
  EXPECT_THROW(node.forward(arg_values, {}), Error);
  const Tensor *cur_value = node.get_inner_values()[0];
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  // backward() updates the gradient of `param`.
  EXPECT_NO_THROW(node.backward(
        arg_values, { cur_value }, { &cur_grad }, arg_grads));
  EXPECT_EQ("Parameter", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(dev, node.get_device());
  EXPECT_TRUE(vector_match(vector<float>(4, 42), cur_value->to_vector()));
  EXPECT_TRUE(vector_match(vector<float>(4, 42), param.value().to_vector()));
  EXPECT_TRUE(vector_match(vector<float>(4, 1), param.gradient().to_vector()));
  EXPECT_EQ(&param.value(), cur_value);
}

TEST_F(OperatorImplTest, CheckCopy) {
  devices::Naive dev2;
  const Shape ret_shape({2, 2}, 3);
  setup_1arg();
  Copy node(dev2);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = dev2.new_tensor_by_vector(
      ret_shape, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  EXPECT_NO_THROW(node.backward(
        arg_values, { &cur_value }, { &cur_grad }, arg_grads));
  EXPECT_EQ("Copy", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(&dev2, node.get_device());
  EXPECT_TRUE(vector_match(arg_values[0]->to_vector(), cur_value.to_vector()));
  EXPECT_TRUE(vector_match(arg_grads[0]->to_vector(), cur_grad.to_vector()));
}

TEST_F(OperatorImplTest, CheckConstant) {
  struct TestCase {
    Shape shape;
    float k;
  };
  const vector<TestCase> test_cases {
    {Shape({}), 0},
    {Shape({2}, 2), 1},
    {Shape({2, 3}, 3), -1},
    {Shape({2, 3, 5}, 4), 42},
    {Shape({2, 3, 5, 7}, 5), -123},
  };
  for (const TestCase &tc : test_cases) {
    Constant node(tc.shape, tc.k, *dev);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.shape, *dev);
    // backward() has no effect.
    EXPECT_NO_THROW(node.backward(
          arg_values, { &cur_value }, { &cur_grad }, arg_grads));
    EXPECT_EQ("Constant(" + std::to_string(tc.k) + ')', node.name());
    EXPECT_EQ(tc.shape, cur_shape);
    EXPECT_EQ(dev, node.get_device());
    EXPECT_TRUE(vector_match(
          vector<float>(tc.shape.size(), tc.k), cur_value.to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckIdentity) {
  struct TestCase {
    std::uint32_t size;
    Shape shape;
    vector<float> data;
  };
  const vector<TestCase> test_cases {
    {1, {}, {1}},
    {2, {2, 2}, {1, 0, 0, 1}},
    {3, {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}},
    {4, {4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
  };
  for (const TestCase &tc : test_cases) {
    Identity node(tc.size, *dev);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.shape, *dev);
    // backward() has no effect.
    EXPECT_NO_THROW(node.backward(
          arg_values, { &cur_value }, { &cur_grad }, arg_grads));
    EXPECT_EQ("Identity(" + std::to_string(tc.size) + ')', node.name());
    EXPECT_EQ(tc.shape, cur_shape);
    EXPECT_EQ(dev, node.get_device());
    EXPECT_TRUE(vector_match(tc.data, cur_value.to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckRandomBernoulli) {
  struct TestCase {
    Shape shape;
    float p;
    vector<float> data;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {Shape({2, 2}, 3), 0.5, {1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0}},
    {Shape({2, 2}, 3), 1, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  };
  for (const TestCase &tc : test_cases) {
    RandomBernoulli node(tc.shape, tc.p, *dev);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.shape, *dev);
    // backward() has no effect.
    EXPECT_NO_THROW(node.backward(
          arg_values, { &cur_value }, { &cur_grad }, arg_grads));
    EXPECT_EQ("RandomBernoulli(" + std::to_string(tc.p) + ')', node.name());
    EXPECT_EQ(tc.shape, cur_shape);
    EXPECT_EQ(dev, node.get_device());
    EXPECT_TRUE(vector_match(tc.data, cur_value.to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckRandomUniform) {
  struct TestCase {
    Shape shape;
    float lower, upper;
    vector<float> data;
  };
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, -1,
      {-1.07038391, -1.10984528, -1.68362451, -1.86929274,
        -1.81608117, -1.96024048, -1.79543972, -1.17356396,
        -1.43227506, -1.46792209, -1.40445530, -1.04368997}},
    {Shape({2, 2}, 3), -1, 1,
      {0.92902899, -0.07614738, 0.30635417, 0.84747636,
        0.49781322, -0.25215763, 0.30713975, -0.69005328,
        0.49542964, 0.78468740, 0.92261350, -0.94642055}},
    {Shape({2, 2}, 3), 1, 2,
      {1.00838828, 1.29150236, 1.10644436, 1.39874411,
        1.29870367, 1.80728865, 1.65641117, 1.62709427,
        1.80981255, 1.90792489, 1.87217593, 1.55639732}},
  };
  for (const TestCase &tc : test_cases) {
    RandomUniform node(tc.shape, tc.lower, tc.upper, *dev);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.shape, *dev);
    // backward() has no effect.
    EXPECT_NO_THROW(node.backward(
          arg_values, { &cur_value }, { &cur_grad }, arg_grads));
    EXPECT_EQ(
        "RandomUniform(" + std::to_string(tc.lower) + ',' +
        std::to_string(tc.upper) + ')', node.name());
    EXPECT_EQ(tc.shape, cur_shape);
    EXPECT_EQ(dev, node.get_device());
    EXPECT_TRUE(vector_match(tc.data, cur_value.to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckRandomNormal) {
  struct TestCase {
    Shape shape;
    float mean, sd;
    vector<float> data;
  };
#ifdef __GLIBCXX__
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, 2,
      {-3.57166052, -2.78148127, -0.94226873, -2.95729542,
        0.35889030, 2.98024511, -0.96429956, -1.78313935,
        -2.08661819, -0.94322312, -0.78637350, -1.56128621}},
    {Shape({2, 2}, 3), 0, 1,
      {-0.69024169, 1.36268508, -0.96791244, 0.43081367,
        0.46228078, 0.29187113, -0.22691579, -0.88196278,
        0.92891538, -0.60850668, 1.20224774, 1.47957015}},
    {Shape({2, 2}, 3), 2, .5,
      {2.07982802, 2.52679920, 2.56787491, 2.26420283,
        2.16808653, 2.08484149, 0.90178788, 1.73830211,
        2.59950328, 1.45240283, 1.47200286, 1.91403091}},
  };
#elif defined _LIBCPP_VERSION
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, 2,
      {-2.78148127, -3.57166052, -2.95729542, -0.94226873,
        2.98024511,  0.35889030, -1.78313935, -0.96429944,
       -0.94322312, -2.08661819, -1.56128621, -0.78637350}},
    {Shape({2, 2}, 3), 0, 1,
      { 1.36268508, -0.69024169,  0.43081367, -0.96791244,
        0.29187110,  0.46228072, -0.88196278, -0.22691579,
       -0.60850668,  0.92891538,  1.47957015,  1.20224774}},
    {Shape({2, 2}, 3), 2, .5,
      {2.52679920, 2.07982802, 2.26420283, 2.56787491,
       2.08484149, 2.16808653, 1.73830211, 0.90178788,
       1.45240283, 2.59950328, 1.91403091, 1.47200286}},
  };
#else
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, 2, {}},
    {Shape({2, 2}, 3), 0, 1, {}},
    {Shape({2, 2}, 3), 2, .5, {}},
  };
  std::cerr << "... Unknown C++ library. Expected results can't be defined." << std::endl;
#endif
  for (const TestCase &tc : test_cases) {
    RandomNormal node(tc.shape, tc.mean, tc.sd, *dev);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.shape, *dev);
    // backward() has no effect.
    EXPECT_NO_THROW(node.backward(
          arg_values, { &cur_value }, { &cur_grad }, arg_grads));
    EXPECT_EQ(
        "RandomNormal(" + std::to_string(tc.mean) + ',' +
        std::to_string(tc.sd) + ')', node.name());
    EXPECT_EQ(tc.shape, cur_shape);
    EXPECT_EQ(dev, node.get_device());
#ifdef PRIMITIV_MAYBE_FPMATH_X87
    EXPECT_TRUE(vector_near(tc.data, cur_value.to_vector(), 1e-6));
#else
    EXPECT_TRUE(vector_match(tc.data, cur_value.to_vector()));
#endif
  }
}

TEST_F(OperatorImplTest, CheckRandomLogNormal) {
  struct TestCase {
    Shape shape;
    float mean, sd;
    vector<float> data;
  };
#ifdef __GLIBCXX__
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, 2,
      {0.02810914, 0.06194668, 0.38974261, 0.05195925,
        1.43173969, 19.69264221, 0.38125014, 0.16810957,
        0.12410613, 0.38937083, 0.45549366, 0.20986597}},
    {Shape({2, 2}, 3), 0, 1,
      {0.50145483, 3.90666890, 0.37987521, 1.53850889,
        1.58769107, 1.33893049, 0.79698789, 0.41396958,
        2.53176165, 0.54416287, 3.32758808, 4.39105797}},
    {Shape({2, 2}, 3), 2, .5,
      {8.00309277, 12.51338959, 13.03808784, 9.62345028,
        8.74154091, 8.04331684, 2.46400452, 5.68767834,
        13.45705223, 4.27337027, 4.35795498, 6.78036499}},
  };
#elif defined _LIBCPP_VERSION
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, 2,
      { 0.06194668, 0.02810914, 0.05195925, 0.38974261,
       19.69264221, 1.43173969, 0.16810957, 0.38125020,
        0.38937083, 0.12410613, 0.20986597, 0.45549366}},
    {Shape({2, 2}, 3), 0, 1,
      {3.90666890, 0.50145483, 1.53850889, 0.37987521,
       1.33893037, 1.58769095, 0.41396958, 0.79698789,
       0.54416287, 2.53176165, 4.39105797, 3.32758808}},
    {Shape({2, 2}, 3), 2, .5,
      {12.51338959,  8.00309277, 9.62345028, 13.03808784,
        8.04331684,  8.74154091, 5.68767834,  2.46400452,
        4.27337027, 13.45705223, 6.78036499,  4.35795498}},
  };
#else
  const vector<TestCase> test_cases {
    {Shape({2, 2}, 3), -2, 2, {}},
    {Shape({2, 2}, 3), 0, 1, {}},
    {Shape({2, 2}, 3), 2, .5, {}},
  };
  std::cerr
    << "... Unknown C++ library. Expected results can't be defined."
    << std::endl;
#endif
  for (const TestCase &tc : test_cases) {
    RandomLogNormal node(tc.shape, tc.mean, tc.sd, *dev);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.shape, *dev);
    // backward() has no effect.
    EXPECT_NO_THROW(node.backward(
          arg_values, { &cur_value }, { &cur_grad }, arg_grads));
    EXPECT_EQ(
        "RandomLogNormal(" + std::to_string(tc.mean) + ',' +
        std::to_string(tc.sd) + ')', node.name());
    EXPECT_EQ(tc.shape, cur_shape);
    EXPECT_EQ(dev, node.get_device());
#ifdef PRIMITIV_MAYBE_FPMATH_X87
    EXPECT_TRUE(vector_near(tc.data, cur_value.to_vector(), 1e-5));
#else
    EXPECT_TRUE(vector_match(tc.data, cur_value.to_vector()));
#endif
  }
}

TEST_F(OperatorImplTest, CheckPick) {
  struct TestCase {
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    Shape ret_shape;
    vector<float> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {0, {0}, Shape({1, 2}, 3),
      {1, 3, 0, 0, -1, -3},
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}},
    {0, {0, 0, 0}, Shape({1, 2}, 3),
      {1, 3, 0, 0, -1, -3},
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}},
    {0, {1, 1, 1}, Shape({1, 2}, 3),
      {2, 4, 0, 0, -2, -4},
      {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
    {1, {0, 0, 1}, Shape({2}, 3),
      {1, 2, 0, 0, -3, -4},
      {1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}},
    {2, {0}, Shape({2, 2}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  };
  setup_1arg();
  for (const TestCase &tc : test_cases) {
    Pick node(tc.ids, tc.dim);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("Pick(" + std::to_string(tc.dim) + ')', node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckSlice) {
  struct TestCase {
    std::uint32_t dim, lower, upper;
    Shape ret_shape;
    vector<float> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {0, 0, 1, Shape({1, 2}, 3),
      {1, 3, 0, 0, -1, -3},
      {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}},
    {0, 1, 2, Shape({1, 2}, 3),
      {2, 4, 0, 0, -2, -4},
      {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
    {0, 0, 2, Shape({2, 2}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {1, 0, 1, Shape({2, 1}, 3),
      {1, 2, 0, 0, -1, -2},
      {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
    {1, 1, 2, Shape({2, 1}, 3),
      {3, 4, 0, 0, -3, -4},
      {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}},
    {1, 0, 2, Shape({2, 2}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {2, 0, 1, Shape({2, 2}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {3, 0, 1, Shape({2, 2}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  };
  setup_1arg();
  for (const TestCase &tc : test_cases) {
    Slice node(tc.dim, tc.lower, tc.upper);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ(
        "Slice(" + std::to_string(tc.dim) + ',' +
        std::to_string(tc.lower) + ':' + std::to_string(tc.upper) + ')',
        node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckSplit) {
  struct TestCase {
    std::uint32_t dim, n;
    Shape ret_shape;
    vector<vector<float>> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {0, 1, Shape({2, 2}, 3),
      {{1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4}},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {0, 2, Shape({1, 2}, 3),
      {{1, 3, 0, 0, -1, -3}, {2, 4, 0, 0, -2, -4}},
      {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2}},
    {1, 1, Shape({2, 2}, 3),
      {{1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4}},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {1, 2, Shape({2}, 3),
      {{1, 2, 0, 0, -1, -2}, {3, 4, 0, 0, -3, -4}},
      {1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2}},
    {2, 1, Shape({2, 2}, 3),
      {{1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4}},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  };
  setup_1arg();
  for (const TestCase &tc : test_cases) {
    Split node(tc.dim, tc.n);
    EXPECT_EQ(1u, node.num_arguments());
    EXPECT_EQ(tc.n, node.num_returns());
    vector<Shape *> cur_shapes;
    vector<Tensor *> cur_values, cur_grads;
    for (std::uint32_t i = 0; i < tc.n; ++i) {
      cur_shapes.emplace_back(new Shape());
      cur_values.emplace_back(new Tensor());
      cur_grads.emplace_back(new Tensor(
            functions::constant<Tensor>(tc.ret_shape, (i + 1), *dev)));
    }
    node.forward_shape(arg_shapes, cur_shapes);
    node.forward(arg_values, cur_values);
    reset_gradients();
    node.backward(
        arg_values,
        vector<const Tensor *>(cur_values.begin(), cur_values.end()),
        vector<const Tensor *>(cur_grads.begin(), cur_grads.end()),
        arg_grads);
    EXPECT_EQ(
        "Split(" + std::to_string(tc.dim) + ',' + std::to_string(tc.n) + ')',
        node.name());
    EXPECT_EQ(nullptr, node.get_device());
    for (std::uint32_t i = 0; i < tc.n; ++i) {
      EXPECT_EQ(tc.ret_shape, *cur_shapes[i]);
      EXPECT_TRUE(vector_match(tc.ret_data[i], cur_values[i]->to_vector()));
    }
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
    for (std::uint32_t i = 0; i < tc.n; ++i) {
      delete cur_shapes[i];
      delete cur_values[i];
      delete cur_grads[i];
    }
  }
}

TEST_F(OperatorImplTest, CheckConcat) {
  struct TestCase {
    std::uint32_t dim;
    Shape ret_shape;
    vector<float> cur_value_data;
    vector<float> cur_grad_data;
  };
  const vector<TestCase> test_cases {
    {0, Shape({4, 2}, 3),
      {1, 2, 1, 1, 3, 4, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2, -1, -2, 3, 3, -3, -4, 3, 3},
      {1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2}},
    {1, Shape({2, 4}, 3),
      {1, 2, 3, 4, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, -1, -2, -3, -4, 3, 3, 3, 3},
      {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2}},
    {2, Shape({2, 2, 2}, 3),
      {1, 2, 3, 4, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, -1, -2, -3, -4, 3, 3, 3, 3},
      {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2}},
    {3, Shape({2, 2, 1, 2}, 3),
      {1, 2, 3, 4, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, -1, -2, -3, -4, 3, 3, 3, 3},
      {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2}},
  };
  setup_2args();
  for (const TestCase &tc : test_cases) {
    Concat node(tc.dim);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = dev->new_tensor_by_vector(
        tc.ret_shape, tc.cur_grad_data);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("Concat(" + std::to_string(tc.dim) + ')', node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.cur_value_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(12, 1), arg_grads[0]->to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(12, 2), arg_grads[1]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckReshape) {
  // y = reshape(x)
  // dy/dx = 1
  setup_1arg();
  const vector<Shape> shapes {
    {4}, {1, 4}, {1, 1, 4},
    {2, 2}, {2, 1, 2}, {1, 2, 2},
    Shape({4}, 3), Shape({1, 4}, 3), Shape({1, 1, 4}, 3),
    Shape({2, 2}, 3), Shape({2, 1, 2}, 3), Shape({1, 2, 2}, 3),
  };
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  for (const Shape ret_shape : shapes) {
    Reshape node(ret_shape);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(
        ret_shape.resize_batch(3), *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("Reshape(" + ret_shape.to_string() + ')', node.name());
    EXPECT_EQ(ret_shape.resize_batch(3), cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(bw_grad, arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckFlatten) {
  // y = flatten(x)
  // dy/dx = 1
  setup_1arg();
  const Shape ret_shape({4}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  TEST_1ARG(Flatten);
}

TEST_F(OperatorImplTest, CheckPositive) {
  // y = x
  // dy/dx = 1
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  TEST_1ARG(Positive);
}

TEST_F(OperatorImplTest, CheckNegative) {
  // y = -x
  // dy/dx = -1
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {-1, -2, -3, -4, 0, 0, 0, 0, 1, 2, 3, 4};
  const vector<float> bw_grad(arg_shapes[0]->size(), -1);
  TEST_1ARG(Negative);
}

TEST_F(OperatorImplTest, CheckAddConst) {
  // y = x + k
  // dy/dx = 1
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {4, 5, 6, 7, 3, 3, 3, 3, 2, 1, 0, -1};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  TEST_1ARG_K(AddConst, 3);
}

TEST_F(OperatorImplTest, CheckSubtractConstR) {
  // y = x - k
  // dy/dx = 1
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {-2, -1, 0, 1, -3, -3, -3, -3, -4, -5, -6, -7};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  TEST_1ARG_K(SubtractConstR, 3);
}

TEST_F(OperatorImplTest, CheckSubtractConstL) {
  // y = k - x
  // dy/dx = -1
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {2, 1, 0, -1, 3, 3, 3, 3, 4, 5, 6, 7};
  const vector<float> bw_grad(arg_shapes[0]->size(), -1);
  TEST_1ARG_K(SubtractConstL, 3);
}

TEST_F(OperatorImplTest, CheckMultiplyConst) {
  // y = kx
  // dy/dx = k
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {3, 6, 9, 12, 0, 0, 0, 0, -3, -6, -9, -12};
  const vector<float> bw_grad(arg_shapes[0]->size(), 3);
  TEST_1ARG_K(MultiplyConst, 3);
}

TEST_F(OperatorImplTest, CheckDivideConstR) {
  // y = x/k
  // dy/dx = 1/k
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1./3, 2./3, 1, 4./3, 0, 0, 0, 0, -1./3, -2./3, -1, -4./3};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1./3);
  TEST_1ARG_K(DivideConstR, 3);
}

TEST_F(OperatorImplTest, CheckDivideConstL) {
  // y = k/x
  // dy/dx = -k/(x^2)
  setup_1arg_nonzero();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    3, 1.5, 1, .75, 3, -3, 3, -3, -3, -1.5, -1, -.75};
  const vector<float> bw_grad {
    -3, -.75, -1./3, -.1875, -3, -3, -3, -3, -3, -.75, -1./3, -.1875};
  TEST_1ARG_K(DivideConstL, 3);
}

TEST_F(OperatorImplTest, CheckPowConstR) {
  // y = x^k
  // dy/dx = k * x^(k-1)
  auto fy = [](float x) { return std::pow(x, 3); };
  auto fgx = [](float x) { return 3 * std::pow(x, 2); };
  setup_1arg_nonnegative();
  const Shape ret_shape({2, 2}, 3);
  vector<float> ret_data, bw_grad;
  const vector<float> x_val = arg_values[0]->to_vector();
  std::transform(x_val.begin(), x_val.end(), std::back_inserter(ret_data), fy);
  std::transform(x_val.begin(), x_val.end(), std::back_inserter(bw_grad), fgx);
  TEST_1ARG_K(PowConstR, 3);
}

TEST_F(OperatorImplTest, CheckPowConstL) {
  // y = k^x
  // dy/dx = log(k) k^x
  auto fy = [](float x) { return std::pow(3, x); };
  auto fgx = [](float x) { return std::log(3) * std::pow(3, x); };
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  vector<float> ret_data, bw_grad;
  const vector<float> x_val = arg_values[0]->to_vector();
  std::transform(x_val.begin(), x_val.end(), std::back_inserter(ret_data), fy);
  std::transform(x_val.begin(), x_val.end(), std::back_inserter(bw_grad), fgx);
  TEST_1ARG_K(PowConstL, 3);
}

TEST_F(OperatorImplTest, CheckAddScalar) {
  // y = x + k
  // dy/dx = 1
  // dy/dk = 1
  setup_2args_scalar();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {2, 3, 4, 5, 2, 2, 2, 2, 2, 1, 0, -1};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), 4),
  };
  TEST_2ARGS(AddScalar);
}

TEST_F(OperatorImplTest, CheckSubtractScalarR) {
  // y = x - k
  // dy/dx = 1
  // dy/dk = -1
  setup_2args_scalar();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {0, 1, 2, 3, -2, -2, -2, -2, -4, -5, -6, -7};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), -4),
  };
  TEST_2ARGS(SubtractScalarR);
}

TEST_F(OperatorImplTest, CheckSubtractScalarL) {
  // y = k - x
  // dy/dx = -1
  // dy/dk = 1
  setup_2args_scalar();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {0, -1, -2, -3, 2, 2, 2, 2, 4, 5, 6, 7};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), -1),
    vector<float>(arg_shapes[1]->size(), 4),
  };
  TEST_2ARGS(SubtractScalarL);
}

TEST_F(OperatorImplTest, CheckMultiplyScalar) {
  // y = kx
  // dy/dx = k
  // dy/dk = x
  setup_2args_scalar();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -3, -6, -9, -12};
  const vector<vector<float>> bw_grads {
    {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
    {10, 0, -10},
  };
  TEST_2ARGS(MultiplyScalar);
}

TEST_F(OperatorImplTest, CheckDivideScalarR) {
  // y = x/k
  // dy/dx = 1/k
  // dy/dk = -x/(k^2)
  setup_2args_scalar();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4, 0, 0, 0, 0, -1./3, -2./3, -1, -4./3,
  };
  const vector<vector<float>> bw_grads {
    {1, 1, 1, 1, .5, .5, .5, .5, 1./3, 1./3, 1./3, 1./3},
    {-10, 0, 10./9},
  };
  TEST_2ARGS(DivideScalarR);
}

TEST_F(OperatorImplTest, CheckDivideScalarL) {
  // y = k/x
  // dy/dx = -k/(x^2)
  // dy/dk = 1/x
  setup_2args_scalar_nonzero();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, .5, 1./3, .25, 2, -2, 2, -2, -3, -1.5, -1, -.75,
  };
  const vector<vector<float>> bw_grads {
    {-1, -.25, -1./9, -.0625, -2, -2, -2, -2, -3, -.75, -1./3, -.1875},
    {1.75 + 1./3, 0, -1.75 - 1./3},
  };
  TEST_2ARGS(DivideScalarL);
}

TEST_F(OperatorImplTest, CheckPowScalarR) {
  // y = x^k
  // dy/dx = k * x^(k-1)
  // dy/dk = log(x) * x^k
  auto fy = [](float x, float k) { return std::pow(x, k); };
  auto fgx = [](float x, float k) { return k * std::pow(x, k - 1); };
  auto fgk = [](float x, float k) { return std::log(x) * std::pow(x, k); };
  setup_2args_scalar_nonnegative();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> x_val = arg_values[0]->to_vector();
  const vector<float> k_val = arg_values[1]->to_vector();
  vector<float> ret_data(ret_shape.size(), 0);
  vector<vector<float>> bw_grads {
    vector<float>(x_val.size(), 0),
    vector<float>(k_val.size(), 0),
  };
  for (std::size_t ix = 0; ix < x_val.size(); ++ix) {
    const std::size_t ik = ix * k_val.size() / x_val.size();
    ret_data[ix] += fy(x_val[ix], k_val[ik]);
    bw_grads[0][ix] += fgx(x_val[ix], k_val[ik]);
    bw_grads[1][ik] += fgk(x_val[ix], k_val[ik]);
  }
  TEST_2ARGS(PowScalarR);
}

TEST_F(OperatorImplTest, CheckPowScalarL) {
  // y = k^x
  // dy/dx = log(k) * k^x
  // dy/dk = x * k^(x-1)
  auto fy = [](float x, float k) { return std::pow(k, x); };
  auto fgx = [](float x, float k) { return std::log(k) * std::pow(k, x); };
  auto fgk = [](float x, float k) { return x * std::pow(k, x - 1); };
  setup_2args_scalar();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> x_val = arg_values[0]->to_vector();
  const vector<float> k_val = arg_values[1]->to_vector();
  vector<float> ret_data(ret_shape.size(), 0);
  vector<vector<float>> bw_grads {
    vector<float>(x_val.size(), 0),
    vector<float>(k_val.size(), 0),
  };
  for (std::size_t ix = 0; ix < x_val.size(); ++ix) {
    const std::size_t ik = ix * k_val.size() / x_val.size();
    ret_data[ix] += fy(x_val[ix], k_val[ik]);
    bw_grads[0][ix] += fgx(x_val[ix], k_val[ik]);
    bw_grads[1][ik] += fgk(x_val[ix], k_val[ik]);
  }
  TEST_2ARGS(PowScalarL);
}

TEST_F(OperatorImplTest, CheckAdd) {
  // y = a + b
  // dy/da = 1
  // dy/db = 1
  setup_2args();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {2, 3, 4, 5, 2, 2, 2, 2, 2, 1, 0, -1};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), 1),
  };
  TEST_2ARGS(Add);
}

TEST_F(OperatorImplTest, CheckSubtract) {
  // y = a - b
  // dy/da = 1
  // dy/db = -1
  setup_2args();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {0, 1, 2, 3, -2, -2, -2, -2, -4, -5, -6, -7};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), -1),
  };
  TEST_2ARGS(Subtract);
}

TEST_F(OperatorImplTest, CheckMultiply) {
  // y = ab
  // dy/da = b
  // dy/db = a
  setup_2args();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -3, -6, -9, -12};
  const vector<vector<float>> bw_grads {
    {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
    {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
  };
  TEST_2ARGS(Multiply);
}

TEST_F(OperatorImplTest, CheckDivide) {
  // y = a/b
  // dy/da = 1/b
  // dy/db = -a/(b^2)
  setup_2args();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4, 0, 0, 0, 0, -1./3, -2./3, -1, -4./3};
  const vector<vector<float>> bw_grads {
    {1, 1, 1, 1, .5f, .5f, .5f, .5f, 1./3, 1./3, 1./3, 1./3},
    {-1, -2, -3, -4, 0, 0, 0, 0, 1./9, 2./9, 1./3, 4./9},
  };
  TEST_2ARGS(Divide);
}

TEST_F(OperatorImplTest, CheckPow) {
  // y = a^b
  // dy/da = b * a^(b-1)
  // dy/db = log(a) a^b
  auto fy = [](float a, float b) { return std::pow(a, b); };
  auto fga = [](float a, float b) { return b * std::pow(a, b - 1); };
  auto fgb = [](float a, float b) { return std::log(a) * std::pow(a, b); };
  setup_2args_swapped();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> a_val = arg_values[0]->to_vector();
  const vector<float> b_val = arg_values[1]->to_vector();
  vector<float> ret_data;
  vector<vector<float>> bw_grads(2);
  for (std::size_t i = 0; i < ret_shape.size(); ++i) {
    ret_data.emplace_back(fy(a_val[i], b_val[i]));
    bw_grads[0].emplace_back(fga(a_val[i], b_val[i]));
    bw_grads[1].emplace_back(fgb(a_val[i], b_val[i]));
  }
  TEST_2ARGS(Pow);
}

TEST_F(OperatorImplTest, CheckTranspose) {
  // y = x^T
  // dy/dx = 1^T
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 3, 2, 4, 0, 0, 0, 0, -1, -3, -2, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  TEST_1ARG(Transpose);
}

TEST_F(OperatorImplTest, CheckPermuteDims) {
  // y = x^T
  // dy/dx = 1^T
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 3, 2, 4, 0, 0, 0, 0, -1, -3, -2, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  PermuteDims node({1, 0});
  EXPECT_EQ("PermuteDims", node.name());
  COMMON_PROC;
  COMMON_CHECK;
}

TEST_F(OperatorImplTest, CheckMatrixMultiply) {
  // y = a . b
  // dy/da = b^T
  // dy/db = a^T
  setup_2args();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {4, 6, 4, 6, 0, 0, 0, 0, -12, -18, -12, -18};
  const vector<vector<float>> bw_grads {
    {2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6},
    {3, 7, 3, 7, 0, 0, 0, 0, -3, -7, -3, -7},
  };
  TEST_2ARGS(MatrixMultiply);
}

TEST_F(OperatorImplTest, CheckAbs) {
  // y = abs(x)
  // dy/dx = sign(x)
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4};
  const vector<float> bw_grad = {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1};
  TEST_1ARG(Abs);
}

TEST_F(OperatorImplTest, CheckSqrt) {
  // y = sqrt(x)
  // dy/dx = 1/(2y)
  setup_1arg_nonnegative();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 1.41421356, 1.73205041, 2, .1, .1, .1, .1, 1, 2, 3, 4,
  };
  const vector<float> bw_grad {
    .5, .5/1.41421356, .5/1.73205041, .25, 5, 5, 5, 5, .5, .25, .5/3, .125,
  };
  TEST_1ARG(Sqrt);
}

TEST_F(OperatorImplTest, CheckExp) {
  // y = exp(x)
  // dy/dx = y
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    2.7182818, 7.3890561, 20.085537, 54.598150,
    1, 1, 1, 1,
    .36787944, .13533528, .049787068, .018315639,
  };
  const vector<float> bw_grad = ret_data;
  TEST_1ARG(Exp);
}

TEST_F(OperatorImplTest, CheckLog) {
  // y = log(x)
  // dy/dx = 1/x
  setup_1arg_nonnegative();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    0, 0.6931472, 1.0986123, 1.3862944,
    -4.6051702, -4.6051702, -4.6051702, -4.6051702,
    0, 1.3862944, 2.1972246, 2.7725887,
  };
  const vector<float> bw_grad {
    1, .5, 1./3, .25,
    100, 100, 100, 100,
    1, .25, 1./9, .0625,
  };
  TEST_1ARG(Log);
}

TEST_F(OperatorImplTest, CheckTanh) {
  // y = tanh(x)
  // dy/dx = 1 - y^2
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    .76159416, .96402758, .99505475, .99932930,
    0, 0, 0, 0,
    -.76159416, -.96402758, -.99505475, -.99932930,
  };
  const vector<float> bw_grad {
    .41997434, .070650825, .0098660372, .0013409507,
    1, 1, 1, 1,
    .41997434, .070650825, .0098660372, .0013409507,
  };
  TEST_1ARG_NEAR(Tanh, 1e-6);
}

TEST_F(OperatorImplTest, CheckSin) {
  // y = sin(x)
  // dy/dx = cos(x)
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    .84147098, .90929743, .14112001, -.75680250,
    0, 0, 0, 0,
    -.84147098, -.90929743, -.14112001, .75680250,
  };
  const vector<float> bw_grad {
    .54030231, -.41614684, -.98999250, -.65364362,
    1, 1, 1, 1,
    .54030231, -.41614684, -.98999250, -.65364362,
  };
  TEST_1ARG(Sin);
}

TEST_F(OperatorImplTest, CheckCos) {
  // y = cos(x)
  // dy/dx = -sin(x)
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    .54030231, -.41614684, -.98999250, -.65364362,
    1, 1, 1, 1,
    .54030231, -.41614684, -.98999250, -.65364362,
  };
  const vector<float> bw_grad {
    -.84147098, -.90929743, -.14112001, .75680250,
    0, 0, 0, 0,
    .84147098, .90929743, .14112001, -.75680250,
  };
  TEST_1ARG(Cos);
}

TEST_F(OperatorImplTest, CheckTan) {
  // y = tan(x)
  // dy/dx = 1 + y^2
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1.5574077, -2.1850399, -.14254654, 1.1578213,
    0, 0, 0, 0,
    -1.5574077, 2.1850399, .14254654, -1.1578213,
  };
  const vector<float> bw_grad {
    3.4255188, 5.7743992, 1.0203195, 2.3405501,
    1, 1, 1, 1,
    3.4255188, 5.7743992, 1.0203195, 2.3405501,
  };
  TEST_1ARG(Tan);
}

TEST_F(OperatorImplTest, CheckSigmoid) {
  // y = sigmoid(x)
  // dy/dx = y * (1 - y)
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    .73105858, .88079708, .95257413, .98201379,
    .5, .5, .5, .5,
    .26894142, .11920292, .047425873, .017986210,
  };
  const vector<float> bw_grad {
    .19661193, .10499359, .045176660, .017662706,
    .25, .25, .25, .25,
    .19661193, .10499359, .045176660, .017662706,
  };
  TEST_1ARG_NEAR(Sigmoid, 1e-6);
}

TEST_F(OperatorImplTest, CheckSoftplus) {
  // y = sigmoid(x)
  // dy/dx = y * (1 - y)
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1.3132617, 2.1269280, 3.0485874, 4.0181499,
    .69314718, .69314718, .69314718, .69314718,
    .31326169, .12692801, .048587352, .018149928,
  };
  const vector<float> bw_grad {
    .73105858, .88079708, .95257413, .98201379,
    .5, .5, .5, .5,
    .26894142, .11920292, .047425873, .017986210,
  };
  TEST_1ARG_NEAR(Softplus, 1e-6);
}

TEST_F(OperatorImplTest, CheckReLU) {
  // y = x > 0 ? x : 0
  // dy/dx = x > 0 ? 1 : 0
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };
  const vector<float> bw_grad {
    1, 1, 1, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };
  TEST_1ARG(ReLU);
}

TEST_F(OperatorImplTest, CheckLReLU) {
  // y = x > 0 ? x : 0.01x
  // dy/dx = x > 0 ? 1 : 0.01
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4,
    0, 0, 0, 0,
    -.01, -.02, -.03, -.04,
  };
  const vector<float> bw_grad {
    1, 1, 1, 1,
    .01, .01, .01, .01,
    .01, .01, .01, .01,
  };
  TEST_1ARG(LReLU);
}

TEST_F(OperatorImplTest, CheckPReLU) {
  // y = x > 0 ? x : ax
  // dy/dx = x > 0 ? 1 : a
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4,
    0, 0, 0, 0,
    -.1, -.2, -.3, -.4,
  };
  const vector<float> bw_grad {
    1, 1, 1, 1,
    .1, .1, .1, .1,
    .1, .1, .1, .1,
  };
  TEST_1ARG_K(PReLU, .1);
}

TEST_F(OperatorImplTest, CheckELU) {
  // y = x > 0 ? x : a * (exp(x) - 1)
  // dy/dx = x > 0 ? 1 : y + a
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4,
    0, 0, 0, 0,
    -.6321206, -.8646647, -.9502129, -.9816844,
  };
  const vector<float> bw_grad {
    1, 1, 1, 1,
    1, 1, 1, 1,
    3.6787944e-01, 1.3533528e-01, 4.9787068e-02, 1.8315639e-02,
  };
  TEST_1ARG_K_NEAR(ELU, 1, 1e-6);
}

TEST_F(OperatorImplTest, CheckFlip) {
  // y = flip(x, dim)
  // dy/dx = flip(1, dim)
  setup_1arg();
  struct TestCase {
    std::uint32_t dim;
    Shape ret_shape;
    vector<float> ret_data;
  };
  const vector<TestCase> test_cases {
    {0, Shape({2, 2}, 3), {2, 1, 4, 3, 0, 0, 0, 0, -2, -1, -4, -3}},
    {1, Shape({2, 2}, 3), {3, 4, 1, 2, 0, 0, 0, 0, -3, -4, -1, -2}},
  };
  for (const TestCase &tc : test_cases) {
    Flip node(tc.dim);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("Flip(" + std::to_string(tc.dim) + ')', node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(12, 1), arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckSum) {
  // y = sum(x, dim)
  // dy/dx = broadcast(1, dim, x.shape[dim])
  setup_1arg();
  struct TestCase {
    std::uint32_t dim;
    Shape ret_shape;
    vector<float> ret_data;
  };
  const vector<TestCase> test_cases {
    {0, Shape({1, 2}, 3), {3, 7, 0, 0, -3, -7}},
    {1, Shape({2}, 3), {4, 6, 0, 0, -4, -6}},
    {2, Shape({2, 2}, 3), {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4}},
  };
  for (const TestCase &tc : test_cases) {
    Sum node(tc.dim);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("Sum(" + std::to_string(tc.dim) + ')', node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(12, 1), arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckLogSumExp) {
  // y = logsumexp(x, dim)
  // dy/dx = softmax(x, dim)
  setup_1arg();
  struct TestCase {
    std::uint32_t dim;
    Shape ret_shape;
    vector<float> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {0, Shape({1, 2}, 3),
      {2.31326169, 4.31326169,
        0.69314718, 0.69314718,
        -0.68673831, -2.68673831},
      {0.26894142, 0.73105858, 0.26894142, 0.73105858,
        .5, .5, .5, .5,
        0.73105858, 0.26894142, 0.73105858, 0.26894142}},
    {1, Shape({2, 1}, 3),
      {3.12692801, 4.12692801,
        0.69314718, 0.69314718,
        -0.87307199, -1.87307199},
      {0.11920292, 0.11920292, 0.88079708, 0.88079708,
        .5, .5, .5, .5,
        0.88079708, 0.88079708, 0.11920292, 0.11920292}},
    {2, Shape({2, 2, 1}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  };
  for (const TestCase &tc : test_cases) {
    LogSumExp node(tc.dim);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("LogSumExp(" + std::to_string(tc.dim) + ')', node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckBroadcast) {
  // y = broadcast(x, dim, size)
  // dy/dx = sum(1, dim)
  setup_1arg();
  struct TestCase {
    std::uint32_t dim, size;
    Shape ret_shape;
    vector<float> ret_data;
  };
  const vector<TestCase> test_cases {
    {2, 2, Shape({2, 2, 2}, 3),
      {1, 2, 3, 4, 1, 2, 3, 4,
        0, 0, 0, 0, 0, 0, 0, 0,
        -1, -2, -3, -4, -1, -2 ,-3, -4,}},
    {3, 2, Shape({2, 2, 1, 2}, 3),
      {1, 2, 3, 4, 1, 2, 3, 4,
        0, 0, 0, 0, 0, 0, 0, 0,
        -1, -2, -3, -4, -1, -2 ,-3, -4,}},
    {2, 3, Shape({2, 2, 3}, 3),
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1, -2, -3, -4, -1, -2 ,-3, -4, -1, -2, -3, -4}},
  };
  for (const TestCase tc : test_cases) {
    Broadcast node(tc.dim, tc.size);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ(
        "Broadcast(" + std::to_string(tc.dim) + ','
        + std::to_string(tc.size) + ')', node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(
        vector_match(vector<float>(12, tc.size), arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckBatchPick) {
  struct TestCase {
    vector<std::uint32_t> ids;
    Shape ret_shape;
    vector<float> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {{0}, Shape({2, 2}, 1),
      {1, 2, 3, 4},
      {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{1, 2}, Shape({2, 2}, 2),
      {0, 0, 0, 0, -1, -2, -3, -4},
      {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}},
    {{0, 0, 0}, Shape({2, 2}, 3),
      {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
      {3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0}},
    {{1, 0, 1, 2}, Shape({2, 2}, 4),
      {0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1}},
    {{2, 1, 0}, Shape({2, 2}, 3),
      {-1, -2, -3, -4, 0, 0, 0, 0, 1, 2, 3, 4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
  };
  setup_1arg();
  for (const TestCase &tc : test_cases) {
    BatchPick node(tc.ids);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ("BatchPick", node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckBatchSlice) {
  struct TestCase {
    std::uint32_t lower, upper;
    Shape ret_shape;
    vector<float> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {0, 1, Shape({2, 2}, 1),
      {1, 2, 3, 4},
      {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
    {0, 2, Shape({2, 2}, 2),
      {1, 2, 3, 4, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}},
    {0, 3, Shape({2, 2}, 3),
      {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {1, 2, Shape({2, 2}, 1),
      {0, 0, 0, 0},
      {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}},
    {1, 3, Shape({2, 2}, 2),
      {0, 0, 0, 0, -1, -2, -3, -4},
      {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}},
    {2, 3, Shape({2, 2}, 1),
      {-1, -2, -3, -4},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}},
  };
  setup_1arg();
  for (const TestCase &tc : test_cases) {
    BatchSlice node(tc.lower, tc.upper);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ(
        "BatchSlice(" +
        std::to_string(tc.lower) + ':' + std::to_string(tc.upper) + ')',
        node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_match(tc.ret_data, cur_value.to_vector()));
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
  }
}

TEST_F(OperatorImplTest, CheckBatchSplit) {
  arg_shapes.emplace_back(new Shape({2, 2}, 4));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4, 5, 6, 7, 8})));
  arg_grads.emplace_back(
      new Tensor(functions::zeros<Tensor>(*arg_shapes[0], *dev)));
  struct TestCase {
    std::uint32_t n;
    Shape ret_shape;
    vector<vector<float>> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {1, Shape({2, 2}, 4),
      {{1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4, 5, 6, 7, 8}},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
    {2, Shape({2, 2}, 2),
      {{1, 2, 3, 4, 0, 0, 0, 0}, {-1, -2, -3, -4, 5, 6, 7, 8}},
      {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}},
    {4, {2, 2},
      {{1, 2, 3, 4}, {0, 0, 0, 0}, {-1, -2, -3, -4}, {5, 6, 7, 8}},
      {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}},
  };
  for (const TestCase &tc : test_cases) {
    BatchSplit node(tc.n);
    EXPECT_EQ(1u, node.num_arguments());
    EXPECT_EQ(tc.n, node.num_returns());
    vector<Shape *> cur_shapes;
    vector<Tensor *> cur_values, cur_grads;
    for (std::uint32_t i = 0; i < tc.n; ++i) {
      cur_shapes.emplace_back(new Shape());
      cur_values.emplace_back(new Tensor());
      cur_grads.emplace_back(new Tensor(
            functions::constant<Tensor>(tc.ret_shape, (i + 1), *dev)));
    }
    node.forward_shape(arg_shapes, cur_shapes);
    node.forward(arg_values, cur_values);
    reset_gradients();
    node.backward(
        arg_values,
        vector<const Tensor *>(cur_values.begin(), cur_values.end()),
        vector<const Tensor *>(cur_grads.begin(), cur_grads.end()),
        arg_grads);
    EXPECT_EQ(
        "BatchSplit(" + std::to_string(tc.n) + ')',
        node.name());
    EXPECT_EQ(nullptr, node.get_device());
    for (std::uint32_t i = 0; i < tc.n; ++i) {
      EXPECT_EQ(tc.ret_shape, *cur_shapes[i]);
      EXPECT_TRUE(vector_match(tc.ret_data[i], cur_values[i]->to_vector()));
    }
    EXPECT_TRUE(vector_match(tc.bw_grad, arg_grads[0]->to_vector()));
    for (std::uint32_t i = 0; i < tc.n; ++i) {
      delete cur_shapes[i];
      delete cur_values[i];
      delete cur_grads[i];
    }
  }
}

TEST_F(OperatorImplTest, CheckBatchConcat) {
  setup_2args();
  arg_shapes.emplace_back(new Shape({2, 2}, 2));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[2], {-2, -2, -2, -2, 5, 6, 7, 8})));
  arg_grads.emplace_back(
      new Tensor(functions::zeros<Tensor>(*arg_shapes[2], *dev)));
  const Shape ret_shape({2, 2}, 8);
  const vector<float> ret_data {
     1,  2,  3,  4,  0,  0,  0,  0, -1, -2, -3, -4,
     1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
    -2, -2, -2, -2,  5,  6,  7,  8,
  };
  const vector<float> grad_data {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
  };
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), 2),
    vector<float>(arg_shapes[2]->size(), 3),
  };

  BatchConcat node;
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = dev->new_tensor_by_vector(ret_shape, grad_data);
  reset_gradients();
  node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
  EXPECT_EQ("BatchConcat", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(nullptr, node.get_device());
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[0], arg_grads[0]->to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[1], arg_grads[1]->to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[2], arg_grads[2]->to_vector()));
}

TEST_F(OperatorImplTest, CheckBatchSum) {
  // y = sum_i x[i]
  // dy/dx = 1 for every minibatch.
  setup_1arg();
  const Shape ret_shape {2, 2};
  const vector<float> ret_data {0, 0, 0, 0};
  const vector<float> bw_grad(12, 1);
  TEST_1ARG(BatchSum);
}

TEST_F(OperatorImplTest, CheckConvolution2D_001111) {
  // y = conv2d(x, w, options)
  // dy/dx = conv2d_bw_w(w, options)
  // dy/dw = conv2d_bw_x(x, options)
  arg_shapes.emplace_back(new Shape({5, 5}, 3));
  arg_shapes.emplace_back(new Shape({2, 2}, 3));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[0], test_utils::make_iota_vector(5 * 5 * 3, 1))));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[1], test_utils::make_iota_vector(2 * 2 * 3, 1))));
  arg_grads.emplace_back(
      new Tensor(functions::ones<Tensor>(*arg_shapes[0], *dev)));
  arg_grads.emplace_back(
      new Tensor(functions::ones<Tensor>(*arg_shapes[1], *dev)));
  const Shape ret_shape({4, 4}, 3);
  const vector<float> ret_data {
    // minibatch 1
     29,  39,  49,  59,
     79,  89,  99, 109,
    129, 139, 149, 159,
    179, 189, 199, 209,
    // minibatch 2
     743,  769,  795,  821,
     873,  899,  925,  951,
    1003, 1029, 1055, 1081,
    1133, 1159, 1185, 1211,
    // minibatch 3
    2257, 2299, 2341, 2383,
    2467, 2509, 2551, 2593,
    2677, 2719, 2761, 2803,
    2887, 2929, 2971, 3013,
  };
  const vector<vector<float>> bw_grads {
    {
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
    },
    {
      // minibatch 1
      257, 241,
      177, 161,
      // minibatch 2
      657, 641,
      577, 561,
      // minibatch 3
      1057, 1041,
       977,  961,
    },
  };

  Convolution2D node(0, 0, 1, 1, 1, 1);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
  EXPECT_EQ("Convolution2D(0,0,1,1,1,1)", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(nullptr, node.get_device());
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[0], arg_grads[0]->to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[1], arg_grads[1]->to_vector()));
}

TEST_F(OperatorImplTest, CheckConvolution2D_112222) {
  // y = conv2d(x, w, options)
  // dy/dx = conv2d_bw_w(w, options)
  // dy/dw = conv2d_bw_x(x, options)
  arg_shapes.emplace_back(new Shape({5, 5}, 3));
  arg_shapes.emplace_back(new Shape({2, 2}, 3));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[0], test_utils::make_iota_vector(5 * 5 * 3, 1))));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[1], test_utils::make_iota_vector(2 * 2 * 3, 1))));
  arg_grads.emplace_back(
      new Tensor(functions::ones<Tensor>(*arg_shapes[0], *dev)));
  arg_grads.emplace_back(
      new Tensor(functions::ones<Tensor>(*arg_shapes[1], *dev)));
  const Shape ret_shape({3, 3}, 3);
  const vector<float> ret_data {
    // minibatch 1
     7,  23, 18,
    38, 108, 74,
    51, 125, 76,
    // minibatch 2
    160, 362, 204,
    434, 966, 536,
    294, 644, 352,
    // minibatch 3
     513, 1101,  590,
    1230, 2624, 1398,
     737, 1563,  828,
  };
  const vector<vector<float>> bw_grads {
    {
      // minibatch 1
      1,  1, 1,  1, 1,
      1, 11, 1, 11, 1,
      1,  1, 1,  1, 1,
      1, 11, 1, 11, 1,
      1,  1, 1,  1, 1,
      // minibatch 2
      1,  1, 1,  1, 1,
      1, 27, 1, 27, 1,
      1,  1, 1,  1, 1,
      1, 27, 1, 27, 1,
      1,  1, 1,  1, 1,
      // minibatch 3
      1,  1, 1,  1, 1,
      1, 43, 1, 43, 1,
      1,  1, 1,  1, 1,
      1, 43, 1, 43, 1,
      1,  1, 1,  1, 1,
    },
    {
      // minibatch 1
      53, 53,
      53, 53,
      // minibatch 2
      153, 153,
      153, 153,
      // minibatch 3
      253, 253,
      253, 253,
    },
  };

  Convolution2D node(1, 1, 2, 2, 2, 2);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
  EXPECT_EQ("Convolution2D(1,1,2,2,2,2)", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(nullptr, node.get_device());
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[0], arg_grads[0]->to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[1], arg_grads[1]->to_vector()));
}

TEST_F(OperatorImplTest, CheckMaxPooling2D_110011) {
  // y = max_pool2d(x, options)
  // dy/dx = max_pool2d_bw(x, options)
  arg_shapes.emplace_back(new Shape({6, 6}, 3));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[0], test_utils::make_iota_vector(6 * 6 * 3, 1))));
  arg_grads.emplace_back(
      new Tensor(functions::ones<Tensor>(*arg_shapes[0], *dev)));
  const Shape ret_shape({6, 6}, 3);
  const vector<float> ret_data = test_utils::make_iota_vector(6 * 6 * 3, 1);
  const vector<vector<float>> bw_grads {
    vector<float>(6 * 6 * 3, 2),
  };

  MaxPooling2D node(1, 1, 0, 0, 1, 1);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
  EXPECT_EQ("MaxPooling2D(1,1,0,0,1,1)", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(nullptr, node.get_device());
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[0], arg_grads[0]->to_vector()));
}

TEST_F(OperatorImplTest, CheckMaxPooling2D_221122) {
  // y = max_pool2d(x, options)
  // dy/dx = max_pool2d_bw(x, options)
  arg_shapes.emplace_back(new Shape({6, 6}, 3));
  arg_values.emplace_back(new Tensor(dev->new_tensor_by_vector(
      *arg_shapes[0], test_utils::make_iota_vector(6 * 6 * 3, 1))));
  arg_grads.emplace_back(
      new Tensor(functions::ones<Tensor>(*arg_shapes[0], *dev)));
  const Shape ret_shape({4, 4}, 3);
  const vector<float> ret_data = {
    // minibatch 1
     1,  3,  5,  6,
    13, 15, 17, 18,
    25, 27, 29, 30,
    31, 33, 35, 36,
    // minibatch 2
    37, 39, 41, 42,
    49, 51, 53, 54,
    61, 63, 65, 66,
    67, 69, 71, 72,
    // minibatch 3
     73,  75,  77,  78,
     85,  87,  89,  90,
     97,  99, 101, 102,
    103, 105, 107, 108,
  };
  const vector<vector<float>> bw_grads {
    {
      // minibatch 1
      2, 1, 2, 1, 2, 2,
      1, 1, 1, 1, 1, 1,
      2, 1, 2, 1, 2, 2,
      1, 1, 1, 1, 1, 1,
      2, 1, 2, 1, 2, 2,
      2, 1, 2, 1, 2, 2,
      // minibatch 2
      2, 1, 2, 1, 2, 2,
      1, 1, 1, 1, 1, 1,
      2, 1, 2, 1, 2, 2,
      1, 1, 1, 1, 1, 1,
      2, 1, 2, 1, 2, 2,
      2, 1, 2, 1, 2, 2,
      // minibatch 3
      2, 1, 2, 1, 2, 2,
      1, 1, 1, 1, 1, 1,
      2, 1, 2, 1, 2, 2,
      1, 1, 1, 1, 1, 1,
      2, 1, 2, 1, 2, 2,
      2, 1, 2, 1, 2, 2,
    },
  };

  MaxPooling2D node(2, 2, 1, 1, 2, 2);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
  EXPECT_EQ("MaxPooling2D(2,2,1,1,2,2)", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(nullptr, node.get_device());
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
  EXPECT_TRUE(vector_match(bw_grads[0], arg_grads[0]->to_vector()));
}

TEST_F(OperatorImplTest, CheckSoftmaxCrossEntropy) {
  // y = softmax_cross_entropy(x, t, dim)
  // dy/dx = softmax(x) - t
  // dy/dt = -log(softmax(x))
  setup_2args_softmax_cross_entropy();
  const Shape ret_shape({1, 2}, 3);
  const vector<float> ret_data {
    1.31326169, 0.31326169, 0.69314718, 0.69314718, 0.31326169, 1.31326169,
  };
  const vector<vector<float>> bw_grads {
    {-0.73105858, 0.73105858, 0.26894142, -0.26894142,
      -.5, .5, .5, -.5,
      -0.26894142, 0.26894142, 0.73105858, -0.73105858},
    {1.31326169, 0.31326169, 1.31326169, 0.31326169,
      0.69314718, 0.69314718, 0.69314718, 0.69314718,
      0.31326169, 1.31326169, 0.31326169, 1.31326169},
  };
  SoftmaxCrossEntropy node(0);
  Shape cur_shape;
  Tensor cur_value;
  node.forward_shape(arg_shapes, { &cur_shape });
  node.forward(arg_values, { &cur_value });
  const Tensor cur_grad = functions::ones<Tensor>(ret_shape, *dev);
  node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
  EXPECT_EQ("SoftmaxCrossEntropy(0)", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_EQ(nullptr, node.get_device());
  EXPECT_TRUE(vector_near(ret_data, cur_value.to_vector(), 1e-6));
  EXPECT_TRUE(vector_near(bw_grads[0], arg_grads[0]->to_vector(), 1e-6));
  EXPECT_TRUE(vector_near(bw_grads[1], arg_grads[1]->to_vector(), 1e-6));
}

TEST_F(OperatorImplTest, CheckSparseSoftmaxCrossEntropy) {
  struct TestCase {
    std::uint32_t dim;
    vector<std::uint32_t> ids;
    Shape ret_shape;
    vector<float> ret_data;
    vector<float> bw_grad;
  };
  const vector<TestCase> test_cases {
    {0, {0}, Shape({1, 2}, 3),
      {1.31326169, 1.31326169, 0.69314718, 0.69314718, 0.31326169, 0.31326169},
      {-0.73105858, 0.73105858, -0.73105858, 0.73105858,
        -0.5, 0.5, -0.5, 0.5,
        -0.26894142, 0.26894142, -0.26894142, 0.26894142}},
    {0, {0, 0, 0}, Shape({1, 2}, 3),
      {1.31326169, 1.31326169, 0.69314718, 0.69314718, 0.31326169, 0.31326169},
      {-0.73105858, 0.73105858, -0.73105858, 0.73105858,
        -0.5, 0.5, -0.5, 0.5,
        -0.26894142, 0.26894142, -0.26894142, 0.26894142}},
    {0, {1, 1, 1}, Shape({1, 2}, 3),
      {0.31326169, 0.31326169, 0.69314718, 0.69314718, 1.31326169, 1.31326169},
      {0.26894142, -0.26894142, 0.26894142, -0.26894142,
        0.5, -0.5, 0.5, -0.5,
        0.73105858, -0.73105858, 0.73105858, -0.73105858}},
    {1, {0, 0, 1}, Shape({2}, 3),
      {2.12692801, 2.12692801, 0.69314718, 0.69314718, 2.12692801, 2.12692801},
      {-0.88079708, -0.88079708, 0.88079708, 0.88079708,
        -0.5, -0.5, 0.5, 0.5,
        0.88079708, 0.88079708, -0.88079708, -0.88079708}},
    {2, {0}, Shape({2, 2}, 3),
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
  };
  setup_1arg();
  for (const TestCase &tc : test_cases) {
    SparseSoftmaxCrossEntropy node(tc.ids, tc.dim);
    Shape cur_shape;
    Tensor cur_value;
    node.forward_shape(arg_shapes, { &cur_shape });
    node.forward(arg_values, { &cur_value });
    const Tensor cur_grad = functions::ones<Tensor>(tc.ret_shape, *dev);
    reset_gradients();
    node.backward(arg_values, { &cur_value }, { &cur_grad }, arg_grads);
    EXPECT_EQ(
        "SparseSoftmaxCrossEntropy(" + std::to_string(tc.dim) + ')',
        node.name());
    EXPECT_EQ(tc.ret_shape, cur_shape);
    EXPECT_EQ(nullptr, node.get_device());
    EXPECT_TRUE(vector_near(tc.ret_data, cur_value.to_vector(), 1e-6));
    EXPECT_TRUE(vector_near(tc.bw_grad, arg_grads[0]->to_vector(), 1e-6));
  }
}

TEST_F(OperatorImplTest, CheckStopGradient) {
  // y = x
  // dy/dx = 0
  setup_1arg();
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1., 2., 3., 4., 0., 0., 0., 0., -1., -2., -3., -4.,
  };
  const vector<float> bw_grad {
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
  };
  TEST_1ARG(StopGradient);
}

}  // namespace operators
}  // namespace primitiv
