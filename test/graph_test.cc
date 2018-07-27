#include <primitiv/config.h>

#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/core/functions.h>
#include <primitiv/core/graph.h>
#include <primitiv/core/initializer_impl.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/operator_impl.h>
#include <primitiv/core/parameter.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class GraphTest : public testing::Test {
protected:
  devices::Naive dev;
  devices::Naive dev2;
};

TEST_F(GraphTest, CheckDefault) {
  EXPECT_THROW(Graph::get_default(), Error);
  {
    Graph g1;
    Graph::set_default(g1);
    EXPECT_EQ(&g1, &Graph::get_default());
    {
      Graph g2;
      Graph::set_default(g2);
      EXPECT_EQ(&g2, &Graph::get_default());
    }
    EXPECT_THROW(Graph::get_default(), Error);
    Graph g3;
    Graph::set_default(g3);
    EXPECT_EQ(&g3, &Graph::get_default());
  }
  EXPECT_THROW(Graph::get_default(), Error);
}

TEST_F(GraphTest, CheckInvalidNode) {
  Node node;
  EXPECT_FALSE(node.valid());
  EXPECT_THROW(node.graph(), Error);
  EXPECT_THROW(node.operator_id(), Error);
  EXPECT_THROW(node.value_id(), Error);
  EXPECT_THROW(node.shape(), Error);
  EXPECT_THROW(node.device(), Error);
  EXPECT_THROW(node.to_float(), Error);
  EXPECT_THROW(node.to_vector(), Error);
  EXPECT_THROW(node.backward(), Error);
}

TEST_F(GraphTest, CheckMoveNode) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  Node x1 = functions::zeros<Node>({2, 2});
  ASSERT_TRUE(x1.valid());
  const std::uint32_t fid = x1.operator_id();
  const std::uint32_t vid = x1.value_id();

  // c-tor
  Node x2 = std::move(x1);
  EXPECT_FALSE(x1.valid());
  EXPECT_TRUE(x2.valid());
  EXPECT_EQ(fid, x2.operator_id());
  EXPECT_EQ(vid, x2.value_id());

  // assignment
  Node x3;
  x3 = std::move(x2);
  EXPECT_FALSE(x2.valid());
  EXPECT_TRUE(x3.valid());
  EXPECT_EQ(fid, x3.operator_id());
  EXPECT_EQ(vid, x3.value_id());
}

TEST_F(GraphTest, CheckMultipleDevices) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data2 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  const vector<float> data3 {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> grad(12, 1);
  const Node x1 = functions::input<Node>(Shape({2, 2}, 3), data1);
  const Node x2 = functions::input<Node>(Shape({2, 2}, 3), data2, dev2);
  const Node x3 = functions::copy(x1, dev2) + x2;
  EXPECT_EQ(Shape({2, 2}, 3), x3.shape());
  EXPECT_EQ(&dev, &x1.device());
  EXPECT_EQ(&dev2, &x2.device());
  EXPECT_EQ(&dev2, &x3.device());
  EXPECT_NO_THROW(g.forward(x3));
  EXPECT_TRUE(vector_match(data1, g.forward(x1).to_vector()));
  EXPECT_TRUE(vector_match(data1, x1.to_vector()));
  EXPECT_TRUE(vector_match(data2, g.forward(x2).to_vector()));
  EXPECT_TRUE(vector_match(data2, x2.to_vector()));
  EXPECT_TRUE(vector_match(data3, g.forward(x3).to_vector()));
  EXPECT_TRUE(vector_match(data3, x3.to_vector()));
#if 0
  EXPECT_NO_THROW(x3.backward());
  EXPECT_TRUE(vector_match(grad, g.get_gradient(x1).to_vector()));
  EXPECT_TRUE(vector_match(grad, g.get_gradient(x2).to_vector()));
  EXPECT_TRUE(vector_match(grad, g.get_gradient(x3).to_vector()));
#endif
}

TEST_F(GraphTest, CheckInvalidMultipleDevices) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  const vector<float> dummy(12);
  const Node x1 = functions::input<Node>(Shape({2, 2}, 3), dummy);
  const Node x2 = functions::input<Node>(Shape({2, 2}, 3), dummy, dev2);
  const Node x3 = x1 + x2;
  EXPECT_THROW(g.forward(x3), Error);
}

TEST_F(GraphTest, CheckClear) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  EXPECT_EQ(0u, g.num_operators());

  {
    functions::input<Node>({}, {1});
    functions::input<Node>({}, {1});
    EXPECT_EQ(2u, g.num_operators());
  }

  g.clear();
  EXPECT_EQ(0u, g.num_operators());

  {
    functions::input<Node>({}, {1});
    functions::input<Node>({}, {1});
    functions::input<Node>({}, {1});
    EXPECT_EQ(3u, g.num_operators());
  }

  g.clear();
  EXPECT_EQ(0u, g.num_operators());

  // Clear empty graph.
  g.clear();
  EXPECT_EQ(0u, g.num_operators());
}

TEST_F(GraphTest, CheckForwardBackward) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data3 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  vector<Node> nodes;
  nodes.emplace_back(functions::input<Node>(Shape({2, 2}, 3), data1));
  nodes.emplace_back(functions::ones<Node>({2, 2}));
  nodes.emplace_back(functions::input<Node>(Shape({2, 2}, 3), data3));
  nodes.emplace_back(nodes[0] + nodes[1]);
  nodes.emplace_back(nodes[1] - nodes[2]);
  nodes.emplace_back(nodes[3] * nodes[4]);
  nodes.emplace_back(nodes[5] + 1);
  nodes.emplace_back(functions::sum(nodes[6], 0));
  nodes.emplace_back(functions::sum(nodes[7], 1));
  nodes.emplace_back(functions::batch::sum(nodes[8]));

  EXPECT_EQ(10u, nodes.size());
  EXPECT_EQ(10u, g.num_operators());

  // Dump the graph to the output log.
  std::cout << g.dump("dot");

  // Check all shapes and devices.
  const vector<Shape> expected_shapes {
    Shape({2, 2}, 3), {2, 2}, Shape({2, 2}, 3),
    Shape({2, 2}, 3), Shape({2, 2}, 3), Shape({2, 2}, 3),
    Shape({2, 2}, 3),
    Shape({1, 2}, 3), Shape({}, 3), {},
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    EXPECT_EQ(expected_shapes[i], nodes[i].shape());
    EXPECT_EQ(&dev, &nodes[i].device());
  }

  g.forward(nodes.back());

  // Check all node values.
  const vector<vector<float>> expected_values {
    {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
    {1, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
    {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5},
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1},
    {2, 3, 4, 5, 0, 0, 0, 0, -2, -3, -4, -5},
    {3, 4, 5, 6, 1, 1, 1, 1, -1, -2, -3, -4},
    {7, 11, 2, 2, -3, -7},
    {18, 4, -10},
    {12},
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    // This forward method has no effect and only returns the reference to the
    // inner value.
    const Tensor &val = g.forward(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val.to_vector()));
    EXPECT_TRUE(vector_match(expected_values[i], nodes[i].to_vector()));
  }

#if 0
  nodes.back().backward();

  // Check all node gradients.
  const vector<vector<float>> expected_grads {
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1}, // n[1] - n[2]
    {6, 9, 12, 15}, // batch_sum(n[0] + 2*n[1] - n[2])
    {-2, -3, -4, -5, -2, -3, -4, -5, -2, -3, -4, -5}, // -n[0] - n[1]
    {1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1}, // n[4]
    {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5}, // n[3]
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 1
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, // 1
    {1, 1, 1, 1, 1, 1}, // 1
    {1, 1, 1}, // 1
    {1}, // 1
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    const Tensor &val = g.get_gradient(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_grads[i], val.to_vector()));
  }
#endif
}

TEST_F(GraphTest, CheckImplicitForwardBackward) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  // Launches the forward calculation from Graph::backward().
  const vector<float> data {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const Node x = functions::input<Node>(Shape({2, 2}, 3), data);
  const Node y = functions::exp(x);
  EXPECT_NO_THROW(x.backward());
  EXPECT_NO_THROW(y.backward());

  // Does not launch the forward calculation,
  // get_inner_values() will be called instead.
  Parameter pw({2, 2}, initializers::Constant(0));
  const Node w = functions::parameter<Node>(pw);
  pw.reset_gradient();
  EXPECT_NO_THROW(w.backward());
}

TEST_F(GraphTest, CheckMultipleBackward) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  Parameter pw({2, 2}, initializers::Constant(0));
  const Node w = functions::parameter<Node>(pw);
  const Node x = functions::input<Node>({2, 2}, {1, 2, 3, 4});
  const Node y = w * x;

  EXPECT_TRUE(vector_match(vector<float> {0, 0, 0, 0}, y.to_vector()));
  pw.reset_gradient();

  y.backward();
  EXPECT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, pw.gradient().to_vector()));

  y.backward();
  EXPECT_TRUE(vector_match(
        vector<float> {2, 4, 6, 8}, pw.gradient().to_vector()));

  y.backward();
  EXPECT_TRUE(vector_match(
        vector<float> {3, 6, 9, 12}, pw.gradient().to_vector()));
}

TEST_F(GraphTest, CheckNonzeroArgs) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  const Node a = functions::input<Node>({}, {1});
  const Node b = functions::input<Node>({}, {2});
  const Node c = functions::input<Node>({}, {3});

  vector<float> y_val;

  Node y = functions::concat({a}, 0);
  EXPECT_EQ(Shape {}, y.shape());
  EXPECT_NO_THROW(y_val = y.to_vector());
  EXPECT_TRUE(vector_match({1}, y_val));

  y = functions::concat({a, b}, 0);
  EXPECT_EQ(Shape {2}, y.shape());
  EXPECT_NO_THROW(y_val = y.to_vector());
  EXPECT_TRUE(vector_match({1, 2}, y_val));

  y = functions::concat({a, b, c}, 0);
  EXPECT_EQ(Shape {3}, y.shape());
  EXPECT_NO_THROW(y_val = y.to_vector());
  EXPECT_TRUE(vector_match({1, 2, 3}, y_val));

  EXPECT_THROW(functions::concat(vector<Node> {}, 0), Error);
}

TEST_F(GraphTest, CheckMultipleReturnValues) {
  Device::set_default(dev);

  Graph g;
  Graph::set_default(g);

  Parameter px({9}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const Node x = functions::parameter<Node>(px);

  vector<Node> ys;
  EXPECT_NO_THROW(ys = functions::split(x, 0, 3));
  EXPECT_TRUE(vector_match(vector<float> {1, 2, 3}, ys[0].to_vector()));
  EXPECT_TRUE(vector_match(vector<float> {4, 5, 6}, ys[1].to_vector()));
  EXPECT_TRUE(vector_match(vector<float> {7, 8, 9}, ys[2].to_vector()));

  px.reset_gradient();

  ys[1].backward();
  EXPECT_TRUE(vector_match(
        vector<float> {0, 0, 0, 1, 1, 1, 0, 0, 0}, px.gradient().to_vector()));

  (ys[0] + ys[1]).backward();
  EXPECT_TRUE(vector_match(
        vector<float> {1, 1, 1, 2, 2, 2, 0, 0, 0}, px.gradient().to_vector()));

  functions::sum(ys).backward();
  EXPECT_TRUE(vector_match(
        vector<float> {2, 2, 2, 3, 3, 3, 1, 1, 1}, px.gradient().to_vector()));

  EXPECT_THROW(functions::split(x, 0, 2), Error);
}

TEST_F(GraphTest, CheckXor) {
  Device::set_default(dev);

  // Solves a 2-dimension XOR problem with 3-layer perceptron.
  // h = tanh(W1.x + b1)
  // y = W2.h + b2
  Parameter w1({2, 2}, {1, -1, 1, -1});
  Parameter b1({2}, {-1, -1});
  Parameter w2({1, 2}, {1, 1});
  Parameter b2({}, {1});

  const vector<float> inputs {1, 1, 1, -1, -1, 1, -1, -1};
  const vector<float> outputs {1, -1, -1, 1};

  Graph g;
  Graph::set_default(g);

  vector<Node> nodes;
  // sources
  nodes.emplace_back(functions::input<Node>(Shape({2}, 4), inputs));
  nodes.emplace_back(functions::parameter<Node>(w1));
  nodes.emplace_back(functions::parameter<Node>(b1));
  nodes.emplace_back(functions::parameter<Node>(w2));
  nodes.emplace_back(functions::parameter<Node>(b2));
  // calculation
  nodes.emplace_back(functions::matmul(nodes[1], nodes[0]));
  nodes.emplace_back(nodes[5] + nodes[2]);
  nodes.emplace_back(functions::tanh(nodes[6]));
  nodes.emplace_back(functions::matmul(nodes[3], nodes[7]));
  nodes.emplace_back(nodes[8] + nodes[4]);
  // losses
  nodes.emplace_back(functions::input<Node>(Shape({}, 4), outputs));
  nodes.emplace_back(nodes[9] - nodes[10]);
  nodes.emplace_back(nodes[11] * nodes[11]);
  nodes.emplace_back(functions::batch::sum(nodes[12]));

  EXPECT_EQ(nodes.size(), g.num_operators());
  std::cout << g.dump("dot");

  g.forward(nodes.back());

  // Check all node values.
  const float h1 = .76159416;  // tanh(1)
  const float h2 = .99505475;  // tanh(3)
  const float h3 = -.23346060;  // tanh(1) - tanh(3)
  const float h4 = -1.5231883;  // -2 * tanh(1)
  const float h5 = .76653940;  // 1 + tanh(1) - tanh(3)
  const float h6 = -.52318831;  // 1 - 2 * tanh(1)
  const float h7 = .47681169;  // 2 - 2 * tanh(1)
  const vector<vector<float>> expected_values {
    {1, 1, 1, -1, -1, 1, -1, -1},
    {1, -1, 1, -1},
    {-1, -1},
    {1, 1},
    {1},
    {2, -2, 0, 0, 0, 0, -2, 2},
    {1, -3, -1, -1, -1, -1, -3, 1},
    {h1, -h2, -h1, -h1, -h1, -h1, -h2, h1},
    {h3, h4, h4, h3},
    {h5, h6, h6, h5},
    {1, -1, -1, 1},
    {h3, h7, h7, h3},
    {h3 * h3, h7 * h7, h7 * h7, h3 * h3},
    {2 * (h3 * h3 + h7 * h7)},
  };
  for (std::uint32_t i = 0; i < nodes.size(); ++i) {
    const Tensor &val = g.forward(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val.to_vector()));
    EXPECT_TRUE(vector_match(expected_values[i], nodes[i].to_vector()));
  }

  // TODO(odashi): add gradient checking.
}

TEST_F(GraphTest, CheckLSTM) {
  Device::set_default(dev);

  // Software-based LSTM implementation with input/forget/output-gates.
  // i = sigmoid(Wix . x + Wih . h + bi)
  // f = sigmoid(Wfx . x + Wfh . h + bf)
  // o = sigmoid(Wox . x + Woh . h + bo)
  // j = tanh(Wjx . x + Wjh . h + bj)
  // cc = f * c + i * j
  // hh = o * tanh(cc)
  Parameter pWix({2, 2}, {.3, .1, .5, .3});
  Parameter pWfx({2, 2}, {.4, .1, .5, .8});
  Parameter pWox({2, 2}, {.5, .9, .9, .7});
  Parameter pWjx({2, 2}, {.2, .6, .9, .3});
  Parameter pWih({2, 2}, {.2, .3, .3, .3});
  Parameter pWfh({2, 2}, {.8, .4, .8, .3});
  Parameter pWoh({2, 2}, {.6, .2, .2, .7});
  Parameter pWjh({2, 2}, {.6, .4, .9, .5});
  Parameter pbi({2}, initializers::Constant(0));
  Parameter pbf({2}, initializers::Constant(0));
  Parameter pbo({2}, initializers::Constant(0));
  Parameter pbj({2}, initializers::Constant(0));

  Graph g;
  Graph::set_default(g);

  namespace batch = functions::batch;
  using functions::matmul;
  using functions::input;
  using functions::parameter;
  using functions::sigmoid;
  using functions::sum;
  using functions::tanh;
  using functions::zeros;

  const Node x = input<Node>(Shape({2}, 2), {2, -2, 0.5, -0.5});
  const Node h = input<Node>(Shape({2}, 2), {-1, 1, -0.5, 0.5});
  const Node c = zeros<Node>({2});
  const Node Wix = parameter<Node>(pWix);
  const Node Wfx = parameter<Node>(pWfx);
  const Node Wox = parameter<Node>(pWox);
  const Node Wjx = parameter<Node>(pWjx);
  const Node Wih = parameter<Node>(pWih);
  const Node Wfh = parameter<Node>(pWfh);
  const Node Woh = parameter<Node>(pWoh);
  const Node Wjh = parameter<Node>(pWjh);
  const Node bi = parameter<Node>(pbi);
  const Node bf = parameter<Node>(pbf);
  const Node bo = parameter<Node>(pbo);
  const Node bj = parameter<Node>(pbj);

  const Node i = sigmoid(matmul(Wix, x) + matmul(Wih, h) + bi);
  const Node f = sigmoid(matmul(Wfx, x) + matmul(Wfh, h) + bf);
  const Node o = sigmoid(matmul(Wox, x) + matmul(Woh, h) + bo);
  const Node j = tanh(matmul(Wjx, x) + matmul(Wjh, h) + bj);
  const Node cc = f * c + i * j;
  const Node hh = o * tanh(cc);

  const Node t = zeros<Node>({2});
  const Node diff = hh - t;
  const Node loss = diff * diff;
  const Node sum_loss = batch::sum(sum(loss, 0));

  EXPECT_EQ(45u, g.num_operators());

  const Tensor loss_tensor = g.forward(loss);
  const Tensor sum_loss_tensor = g.forward(sum_loss);
  sum_loss.backward();

  const vector<float> expected_losses {
    5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03
  };
  const float expected_sum_loss = std::accumulate(
      begin(expected_losses), end(expected_losses), .0f);

  EXPECT_TRUE(vector_near(expected_losses, loss_tensor.to_vector(), 1e-6));
  EXPECT_TRUE(vector_near(expected_losses, loss.to_vector(), 1e-6));
  EXPECT_FLOAT_EQ(expected_sum_loss, sum_loss_tensor.to_float());
  EXPECT_FLOAT_EQ(expected_sum_loss, sum_loss.to_float());

  auto print = [](const std::string &name, const Tensor &value) {
    std::cout << name << ": shape=" << value.shape().to_string()
      << ", values=[";
    const vector<float> data = value.to_vector();
    for (std::uint32_t i = 0; i < data.size(); ++i) {
      if (i > 0) std::cout << ',';
      std::cout << data[i];
    }
    std::cout << ']' << std::endl;
  };

  std::cout << "VALUES:" << std::endl;
#define PRINT_VALUE(node) print(#node, g.forward(node))
  PRINT_VALUE(x); PRINT_VALUE(h); PRINT_VALUE(c);
  PRINT_VALUE(Wix); PRINT_VALUE(Wfx); PRINT_VALUE(Wox); PRINT_VALUE(Wjx);
  PRINT_VALUE(Wih); PRINT_VALUE(Wfh); PRINT_VALUE(Woh); PRINT_VALUE(Wjh);
  PRINT_VALUE(bi); PRINT_VALUE(bf); PRINT_VALUE(bo); PRINT_VALUE(bj);
  PRINT_VALUE(i); PRINT_VALUE(f); PRINT_VALUE(o); PRINT_VALUE(j);
  PRINT_VALUE(cc); PRINT_VALUE(hh);
  PRINT_VALUE(t); PRINT_VALUE(diff); PRINT_VALUE(loss);
#undef PRINT_VALUE

#if 0
  std::cout << "GRADIENTS:" << std::endl;
#define PRINT_GRAD(node) print(#node, g.get_gradient(node))
  PRINT_GRAD(x); PRINT_GRAD(h); PRINT_GRAD(c);
  PRINT_GRAD(Wix); PRINT_GRAD(Wfx); PRINT_GRAD(Wox); PRINT_GRAD(Wjx);
  PRINT_GRAD(Wih); PRINT_GRAD(Wfh); PRINT_GRAD(Woh); PRINT_GRAD(Wjh);
  PRINT_GRAD(bi); PRINT_GRAD(bf); PRINT_GRAD(bo); PRINT_GRAD(bj);
  PRINT_GRAD(i); PRINT_GRAD(f); PRINT_GRAD(o); PRINT_GRAD(j);
  PRINT_GRAD(cc); PRINT_GRAD(hh);
  PRINT_GRAD(t); PRINT_GRAD(diff); PRINT_GRAD(loss);
#undef PRINT_GRAD
#endif
}

TEST_F(GraphTest, CheckConcatLSTM) {
  Device::set_default(dev);

  // Another implementation of LSTM that concatenates all gates and inputs.
  // All values and gradients should be same as that of "CheckLSTM".
  Parameter pWx({8, 2}, {
      .3, .1, .4, .1, .5, .9, .2, .6,
      .5, .3, .5, .8, .9, .7, .9, .3});
  Parameter pWh({8, 2}, {
      .2, .3, .8, .4, .6, .2, .6, .4,
      .3, .3, .8, .3, .2, .7, .9, .5});
  Parameter pb({8}, initializers::Constant(0));

  Graph g;
  Graph::set_default(g);

  namespace batch = functions::batch;
  using functions::matmul;
  using functions::input;
  using functions::parameter;
  using functions::sigmoid;
  using functions::slice;
  using functions::sum;
  using functions::tanh;
  using functions::zeros;

  const Node x = input<Node>(Shape({2}, 2), {2, -2, 0.5, -0.5});
  const Node h = input<Node>(Shape({2}, 2), {-1, 1, -0.5, 0.5});
  const Node c = zeros<Node>({2});
  const Node Wx = parameter<Node>(pWx);
  const Node Wh = parameter<Node>(pWh);
  const Node b = parameter<Node>(pb);

  const Node u = matmul(Wx, x) + matmul(Wh, h) + b;
  const Node i = sigmoid(slice(u, 0, 0, 2));
  const Node f = sigmoid(slice(u, 0, 2, 4));
  const Node o = sigmoid(slice(u, 0, 4, 6));
  const Node j = tanh(slice(u, 0, 6, 8));
  const Node cc = f * c + i * j;
  const Node hh = o * tanh(cc);

  const Node t = zeros<Node>({2});
  const Node diff = hh - t;
  const Node loss = diff * diff;
  const Node sum_loss = batch::sum(sum(loss, 0));

  EXPECT_EQ(28u, g.num_operators());

  const Tensor loss_tensor = g.forward(loss);
  const Tensor sum_loss_tensor = g.forward(sum_loss);
  sum_loss.backward();

  const vector<float> expected_losses {
    5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03
  };
  const float expected_sum_loss = std::accumulate(
      begin(expected_losses), end(expected_losses), .0f);

  EXPECT_TRUE(vector_near(expected_losses, loss_tensor.to_vector(), 1e-6));
  EXPECT_TRUE(vector_near(expected_losses, loss.to_vector(), 1e-6));
  EXPECT_FLOAT_EQ(expected_sum_loss, sum_loss_tensor.to_float());
  EXPECT_FLOAT_EQ(expected_sum_loss, sum_loss.to_float());

  auto print = [](const std::string &name, const Tensor &value) {
    std::cout << name << ": shape=" << value.shape().to_string()
      << ", values=[";
    const vector<float> data = value.to_vector();
    for (std::uint32_t i = 0; i < data.size(); ++i) {
      if (i > 0) std::cout << ',';
      std::cout << data[i];
    }
    std::cout << ']' << std::endl;
  };

  std::cout << "VALUES:" << std::endl;
#define PRINT_VALUE(node) print(#node, g.forward(node))
  PRINT_VALUE(x); PRINT_VALUE(h); PRINT_VALUE(c);
  PRINT_VALUE(Wx); PRINT_VALUE(Wh); PRINT_VALUE(b);
  PRINT_VALUE(i); PRINT_VALUE(f); PRINT_VALUE(o); PRINT_VALUE(j);
  PRINT_VALUE(cc); PRINT_VALUE(hh);
  PRINT_VALUE(t); PRINT_VALUE(diff); PRINT_VALUE(loss);
#undef PRINT_VALUE

#if 0
  std::cout << "GRADIENTS:" << std::endl;
#define PRINT_GRAD(node) print(#node, g.get_gradient(node))
  PRINT_GRAD(x); PRINT_GRAD(h); PRINT_GRAD(c);
  PRINT_GRAD(Wx); PRINT_GRAD(Wh); PRINT_GRAD(b);
  PRINT_GRAD(i); PRINT_GRAD(f); PRINT_GRAD(o); PRINT_GRAD(j);
  PRINT_GRAD(cc); PRINT_GRAD(hh);
  PRINT_GRAD(t); PRINT_GRAD(diff); PRINT_GRAD(loss);
#undef PRINT_GRAD
#endif
}

}  // namespace primitiv
