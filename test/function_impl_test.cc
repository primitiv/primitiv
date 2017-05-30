#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/function_impl.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {
namespace functions {

class FunctionImplTest_0Arg : public testing::Test {
protected:
  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

class FunctionImplTest_1Arg : public testing::Test {
protected:
  virtual void SetUp() override {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_grads.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[0], vector<float>(arg_shapes[0]->size()))));
  }

  virtual void TearDown() override {
    for (const Shape *x : arg_shapes) delete x;
    for (const Tensor *x : arg_values) delete x;
    for (Tensor *x : arg_grads) delete x;
  }

  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

class FunctionImplTest_1Arg_NonZero : public testing::Test {
protected:
  virtual void SetUp() override {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[0], {1, 2, 3, 4, 1, -1, 1, -1, -1, -2, -3, -4})));
    arg_grads.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[0], vector<float>(arg_shapes[0]->size()))));
  }

  virtual void TearDown() override {
    for (const Shape *x : arg_shapes) delete x;
    for (const Tensor *x : arg_values) delete x;
    for (Tensor *x : arg_grads) delete x;
  }

  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

class FunctionImplTest_2Args : public testing::Test {
protected:
  virtual void SetUp() override {
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_shapes.emplace_back(new Shape({2, 2}, 3));
    arg_values.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[0], {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4})));
    arg_values.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[1], {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3})));
    arg_grads.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[0], vector<float>(arg_shapes[0]->size()))));
    arg_grads.emplace_back(new Tensor(dev.new_tensor(
        *arg_shapes[1], vector<float>(arg_shapes[1]->size()))));
  }

  virtual void TearDown() override {
    for (const Shape *x : arg_shapes) delete x;
    for (const Tensor *x : arg_values) delete x;
    for (Tensor *x : arg_grads) delete x;
  }

  CPUDevice dev;
  vector<const Shape *> arg_shapes;
  vector<const Tensor *> arg_values;
  vector<Tensor *> arg_grads;
};

#define TEST_1ARG(name_) { \
  const Shape cur_shape = node.forward_shape(arg_shapes); \
  const Tensor cur_value = node.forward(arg_values); \
  const Tensor cur_grad = dev.new_tensor(ret_shape, 1); \
  node.backward(cur_value, cur_grad, arg_values, arg_grads); \
  EXPECT_EQ(#name_, node.name()); \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector())); \
  EXPECT_TRUE(vector_match(bw_grad, arg_grads[0]->to_vector())); \
}

#define TEST_1ARG_NEAR(name_, err) { \
  const Shape cur_shape = node.forward_shape(arg_shapes); \
  const Tensor cur_value = node.forward(arg_values); \
  const Tensor cur_grad = dev.new_tensor(ret_shape, 1); \
  node.backward(cur_value, cur_grad, arg_values, arg_grads); \
  EXPECT_EQ(#name_, node.name()); \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_TRUE(vector_near(ret_data, cur_value.to_vector(), err)); \
  EXPECT_TRUE(vector_near(bw_grad, arg_grads[0]->to_vector(), err)); \
}

#define TEST_2ARGS(name_) { \
  const Shape cur_shape = node.forward_shape(arg_shapes); \
  const Tensor cur_value = node.forward(arg_values); \
  const Tensor cur_grad = dev.new_tensor(ret_shape, 1); \
  node.backward(cur_value, cur_grad, arg_values, arg_grads); \
  EXPECT_EQ(#name_, node.name()); \
  EXPECT_EQ(ret_shape, cur_shape); \
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector())); \
  EXPECT_TRUE( \
      vector_match(bw_grads[0], arg_grads[0]->to_vector())); \
  EXPECT_TRUE( \
      vector_match(bw_grads[1], arg_grads[1]->to_vector())); \
}

TEST_F(FunctionImplTest_0Arg, CheckInput) {
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  const Input node(ret_shape, &dev, ret_data);
  const Shape cur_shape = node.forward_shape(arg_shapes);
  const Tensor cur_value = node.forward(arg_values);
  const Tensor cur_grad = dev.new_tensor(ret_shape, 1);
  // backward() has no effect.
  EXPECT_NO_THROW(node.backward(cur_value, cur_grad, arg_values, arg_grads));
  EXPECT_EQ("Input", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_TRUE(vector_match(ret_data, cur_value.to_vector()));
}

TEST_F(FunctionImplTest_0Arg, CheckParameterInput) {
  const Shape ret_shape {2, 2};
  const initializers::Constant init(42);
  Parameter param(ret_shape, &dev);
  param.reset_value(init);
  param.reset_gradient();
  ASSERT_TRUE(vector_match(vector<float>(4, 42), param.value().to_vector()));
  ASSERT_TRUE(vector_match(vector<float>(4, 0), param.gradient().to_vector()));

  const ParameterInput node(&param);
  const Shape cur_shape = node.forward_shape(arg_shapes);
  const Tensor cur_value = node.forward(arg_values);
  const Tensor cur_grad = dev.new_tensor(ret_shape, 1);
  // backward() updates the gradient of `param`.
  EXPECT_NO_THROW(node.backward(cur_value, cur_grad, arg_values, arg_grads));
  EXPECT_EQ("ParameterInput", node.name());
  EXPECT_EQ(ret_shape, cur_shape);
  EXPECT_TRUE(vector_match(vector<float>(4, 42), cur_value.to_vector()));
  EXPECT_TRUE(vector_match(vector<float>(4, 42), param.value().to_vector()));
  EXPECT_TRUE(vector_match(vector<float>(4, 1), param.gradient().to_vector()));
}

TEST_F(FunctionImplTest_1Arg, CheckPositive) {
  // y = x
  // dy/dx = 1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  const Positive node;
  TEST_1ARG(Positive);
}

TEST_F(FunctionImplTest_1Arg, CheckNegative) {
  // y = -x
  // dy/dx = -1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {-1, -2, -3, -4, 0, 0, 0, 0, 1, 2, 3, 4};
  const vector<float> bw_grad(arg_shapes[0]->size(), -1);
  const Negative node;
  TEST_1ARG(Negative);
}

TEST_F(FunctionImplTest_1Arg, CheckAddConst) {
  // y = x + k
  // dy/dx = 1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {4, 5, 6, 7, 3, 3, 3, 3, 2, 1, 0, -1};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  const AddConst node(3);
  TEST_1ARG(AddConst);
}

TEST_F(FunctionImplTest_1Arg, CheckSubtractConstL) {
  // y = k - x
  // dy/dx = -1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {2, 1, 0, -1, 3, 3, 3, 3, 4, 5, 6, 7};
  const vector<float> bw_grad(arg_shapes[0]->size(), -1);
  const SubtractConstL node(3);
  TEST_1ARG(SubtractConstL);
}

TEST_F(FunctionImplTest_1Arg, CheckSubtractConstR) {
  // y = x - k
  // dy/dx = 1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {-2, -1, 0, 1, -3, -3, -3, -3, -4, -5, -6, -7};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  const SubtractConstR node(3);
  TEST_1ARG(SubtractConstR);
}

TEST_F(FunctionImplTest_1Arg, CheckMultiplyConst) {
  // y = kx
  // dy/dx = k
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {3, 6, 9, 12, 0, 0, 0, 0, -3, -6, -9, -12};
  const vector<float> bw_grad(arg_shapes[0]->size(), 3);
  const MultiplyConst node(3);
  TEST_1ARG(MultiplyConst);
}

TEST_F(FunctionImplTest_1Arg_NonZero, CheckDivideConstL) {
  // y = k/x
  // dy/dx = -k/(x^2)
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    3, 1.5, 1, .75, 3, -3, 3, -3, -3, -1.5, -1, -.75};
  const vector<float> bw_grad {
    -3, -.75, -1./3, -.1875, -3, -3, -3, -3, -3, -.75, -1./3, -.1875};
  const DivideConstL node(3);
  TEST_1ARG(DivideConstL);
}

TEST_F(FunctionImplTest_1Arg, CheckDivideConstR) {
  // y = x/k
  // dy/dx = 1/k
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1./3, 2./3, 1, 4./3, 0, 0, 0, 0, -1./3, -2./3, -1, -4./3};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1./3);
  const DivideConstR node(3);
  TEST_1ARG(DivideConstR);
}

TEST_F(FunctionImplTest_1Arg, CheckTranspose) {
  // y = x^T
  // dy/dx = 1^T
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 3, 2, 4, 0, 0, 0, 0, -1, -3, -2, -4};
  const vector<float> bw_grad(arg_shapes[0]->size(), 1);
  const Transpose node;
  TEST_1ARG(Transpose);
}

TEST_F(FunctionImplTest_2Args, CheckAdd) {
  // y = a + b
  // dy/da = 1
  // dy/db = 1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {2, 3, 4, 5, 2, 2, 2, 2, 2, 1, 0, -1};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), 1),
  };
  const Add node;
  TEST_2ARGS(Add);
}

TEST_F(FunctionImplTest_2Args, CheckSubtract) {
  // y = a - b
  // dy/da = 1
  // dy/db = -1
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {0, 1, 2, 3, -2, -2, -2, -2, -4, -5, -6, -7};
  const vector<vector<float>> bw_grads {
    vector<float>(arg_shapes[0]->size(), 1),
    vector<float>(arg_shapes[1]->size(), -1),
  };
  const Subtract node;
  TEST_2ARGS(Subtract);
}

TEST_F(FunctionImplTest_2Args, CheckMultiply) {
  // y = ab
  // dy/da = b
  // dy/db = a
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {1, 2, 3, 4, 0, 0, 0, 0, -3, -6, -9, -12};
  const vector<vector<float>> bw_grads {
    vector<float> {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
    vector<float> {1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4},
  };
  const Multiply node;
  TEST_2ARGS(Multiply);
}

TEST_F(FunctionImplTest_2Args, CheckDivide) {
  // y = a/b
  // dy/da = 1/b
  // dy/db = -a/(b^2)
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    1, 2, 3, 4, 0, 0, 0, 0, -1./3, -2./3, -1, -4./3};
  const vector<vector<float>> bw_grads {
    vector<float> {1, 1, 1, 1, .5f, .5f, .5f, .5f, 1./3, 1./3, 1./3, 1./3},
    vector<float> {-1, -2, -3, -4, 0, 0, 0, 0, 1./9, 2./9, 1./3, 4./9},
  };
  const Divide node;
  TEST_2ARGS(Divide);
}

TEST_F(FunctionImplTest_2Args, CheckDot) {
  // y = a . b
  // dy/da = b^T
  // dy/db = a^T
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {4, 6, 4, 6, 0, 0, 0, 0, -12, -18, -12, -18};
  const vector<vector<float>> bw_grads {
    vector<float> {2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6},
    vector<float> {3, 7, 3, 7, 0, 0, 0, 0, -3, -7, -3, -7},
  };
  const Dot node;
  TEST_2ARGS(Dot);
}

TEST_F(FunctionImplTest_1Arg, CheckExp) {
  // y = exp(x)
  // dy/dx = y
  const Shape ret_shape({2, 2}, 3);
  const vector<float> ret_data {
    2.7182818, 7.3890561, 20.085537, 54.598150,
    1, 1, 1, 1,
    .36787944, .13533528, .049787068, .018315639,
  };
  const vector<float> bw_grad = ret_data;
  const Exp node;
  TEST_1ARG(Exp);
}

TEST_F(FunctionImplTest_1Arg, CheckTanh) {
  // y = tanh(x)
  // dy/dx = 1 - y^2
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
  const Tanh node;
  TEST_1ARG_NEAR(Tanh, 1e-6);
}

TEST_F(FunctionImplTest_1Arg, CheckSigmoid) {
  // y = sigmoid(x)
  // dy/dx = y * (1 - y)
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
  const Sigmoid node;
  TEST_1ARG_NEAR(Sigmoid, 1e-6);
}

TEST_F(FunctionImplTest_1Arg, CheckReLU) {
  // y = max(x, 0)
  // dy/dx = x >= 0 ? 1 : 0
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
  const ReLU node;
  TEST_1ARG(ReLU);
}

TEST_F(FunctionImplTest_1Arg, CheckBatchSum) {
  // y = sum_i x[i]
  // dy/dx = 1 for every minibatch.
  const Shape ret_shape({2, 2});
  const vector<float> ret_data {0, 0, 0, 0};
  const vector<float> bw_grad {
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
  };
  const BatchSum node;
  TEST_1ARG(BatchSum);
}

}  // namespace functions
}  // namespace primitiv
