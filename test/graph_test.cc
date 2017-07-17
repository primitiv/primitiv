#include <config.h>

#include <sstream>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/node_ops.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {

class GraphTest : public testing::Test {
protected:
  CPUDevice dev;
  CPUDevice dev2;
};

TEST_F(GraphTest, CheckDefaultGraph) {
  EXPECT_THROW(Graph::get_default_graph(), Error);
  {
    Graph g;
    Graph::set_default_graph(g);
    EXPECT_EQ(&g, &Graph::get_default_graph());
  }
  EXPECT_THROW(Graph::get_default_graph(), Error);
  {
    Graph g1;
    Graph::set_default_graph(g1);
    EXPECT_EQ(&g1, &Graph::get_default_graph());
    Graph g2;
    Graph::set_default_graph(g2);
    EXPECT_EQ(&g2, &Graph::get_default_graph());
  }
  EXPECT_THROW(Graph::get_default_graph(), Error);
}

TEST_F(GraphTest, CheckMultipleDevices) {
  Graph g;
  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data2 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  const vector<float> data3 {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> grad(12, 1);
  const Node x1 = node_ops::input(Shape({2, 2}, 3), data1, &dev, &g);
  const Node x2 = node_ops::input(Shape({2, 2}, 3), data2, &dev2, &g);
  const Node x3 = node_ops::copy(x1, &dev2) + x2;
  EXPECT_EQ(Shape({2, 2}, 3), x3.shape());
  EXPECT_EQ(&dev, x1.device());
  EXPECT_EQ(&dev2, x2.device());
  EXPECT_EQ(&dev2, x3.device());
  EXPECT_NO_THROW(g.forward(x3));
  EXPECT_TRUE(vector_match(data1, g.get_value(x1).to_vector()));
  EXPECT_TRUE(vector_match(data2, g.get_value(x2).to_vector()));
  EXPECT_TRUE(vector_match(data3, g.get_value(x3).to_vector()));
  EXPECT_NO_THROW(g.backward(x3));
  EXPECT_TRUE(vector_match(grad, g.get_gradient(x1).to_vector()));
  EXPECT_TRUE(vector_match(grad, g.get_gradient(x2).to_vector()));
  EXPECT_TRUE(vector_match(grad, g.get_gradient(x3).to_vector()));
}

TEST_F(GraphTest, CheckInvalidMultipleDevices) {
  Graph g;
  const vector<float> dummy(12);
  const Node x1 = node_ops::input(Shape({2, 2}, 3), dummy, &dev, &g);
  const Node x2 = node_ops::input(Shape({2, 2}, 3), dummy, &dev2, &g);
  const Node x3 = x1 + x2;
  EXPECT_THROW(g.forward(x3), Error);
}

TEST_F(GraphTest, CheckForwardBackward) {
  Graph g;
  const vector<float> data1 {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  const vector<float> data3 {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  vector<Node> nodes;
  nodes.emplace_back(node_ops::input(Shape({2, 2}, 3), data1, &dev, &g));
  nodes.emplace_back(node_ops::ones({2, 2}, &dev, &g));
  nodes.emplace_back(node_ops::input(Shape({2, 2}, 3), data3, &dev, &g));
  nodes.emplace_back(nodes[0] + nodes[1]);
  nodes.emplace_back(nodes[1] - nodes[2]);
  nodes.emplace_back(nodes[3] * nodes[4]);
  nodes.emplace_back(nodes[5] + 1);
  nodes.emplace_back(node_ops::sum(nodes[6], 0));
  nodes.emplace_back(node_ops::sum(nodes[7], 1));
  nodes.emplace_back(node_ops::batch::sum(nodes[8]));

  EXPECT_EQ(10u, nodes.size());
  EXPECT_EQ(10u, g.num_functions());

  // Dump the graph to the output log.
  g.dump();

  // Check all shapes and devices.
  const vector<Shape> expected_shapes {
    Shape({2, 2}, 3), {2, 2}, Shape({2, 2}, 3),
    Shape({2, 2}, 3), Shape({2, 2}, 3), Shape({2, 2}, 3),
    Shape({2, 2}, 3),
    Shape({1, 2}, 3), Shape({}, 3), {},
  };
  for (unsigned i = 0; i < nodes.size(); ++i) {
    EXPECT_EQ(expected_shapes[i], nodes[i].shape());
    EXPECT_EQ(&dev, nodes[i].device());
  }

  // Check all node values are still invalid.
  for (const Node &node : nodes) {
    EXPECT_THROW(g.get_value(node), Error);
    EXPECT_THROW(g.get_gradient(node), Error);
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
  for (unsigned i = 0; i < nodes.size(); ++i) {
    // This forward method has no effect and only returns the reference to the
    // inner value.
    const Tensor &val1 = g.forward(nodes[i]);
    const Tensor &val2 = g.get_value(nodes[i]);
    EXPECT_EQ(&val1, &val2);
    ASSERT_TRUE(val1.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val1.to_vector()));

    // Gradients are also initialized.
    EXPECT_NO_THROW(g.get_gradient(nodes[i]));
  }

  g.backward(nodes.back());

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
  for (unsigned i = 0; i < nodes.size(); ++i) {
    const Tensor &val = g.get_gradient(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_grads[i], val.to_vector()));
  }
}

TEST_F(GraphTest, CheckXor) {
  // Solves a 2-dimension XOR problem with 3-layer perceptron.
  // h = tanh(W1.x + b1)
  // y = W2.h + b2
  Parameter w1("w1", {2, 2}, {1, -1, 1, -1}, &dev);
  Parameter b1("b1", {2}, {-1, -1}, &dev);
  Parameter w2("w2", {1, 2}, {1, 1}, &dev);
  Parameter b2("b2", {}, {1}, &dev);

  const vector<float> inputs {1, 1, 1, -1, -1, 1, -1, -1};
  const vector<float> outputs {1, -1, -1, 1};

  Graph g;
  vector<Node> nodes;
  // sources
  nodes.emplace_back(node_ops::input(Shape({2}, 4), inputs, &dev, &g));
  nodes.emplace_back(node_ops::input(&w1, &g));
  nodes.emplace_back(node_ops::input(&b1, &g));
  nodes.emplace_back(node_ops::input(&w2, &g));
  nodes.emplace_back(node_ops::input(&b2, &g));
  // calculation
  nodes.emplace_back(node_ops::matmul(nodes[1], nodes[0]));
  nodes.emplace_back(nodes[5] + nodes[2]);
  nodes.emplace_back(node_ops::tanh(nodes[6]));
  nodes.emplace_back(node_ops::matmul(nodes[3], nodes[7]));
  nodes.emplace_back(nodes[8] + nodes[4]);
  // losses
  nodes.emplace_back(node_ops::input(Shape({}, 4), outputs, &dev, &g));
  nodes.emplace_back(nodes[9] - nodes[10]);
  nodes.emplace_back(nodes[11] * nodes[11]);
  nodes.emplace_back(node_ops::batch::sum(nodes[12]));

  EXPECT_EQ(nodes.size(), g.num_functions());
  g.dump();

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
  for (unsigned i = 0; i < nodes.size(); ++i) {
    const Tensor &val = g.get_value(nodes[i]);
    ASSERT_TRUE(val.valid());
    EXPECT_TRUE(vector_match(expected_values[i], val.to_vector()));
  }

  // TODO(odashi): add gradient checking.
}

TEST_F(GraphTest, CheckLSTM) {
  // Software-based LSTM implementation with input/forget/output-gates.
  // i = sigmoid(Wix . x + Wih . h + bi)
  // f = sigmoid(Wfx . x + Wfh . h + bf)
  // o = sigmoid(Wox . x + Woh . h + bo)
  // j = tanh(Wjx . x + Wjh . h + bj)
  // cc = f * c + i * j
  // hh = o * tanh(cc)
  Parameter pWix("Wix", {2, 2}, {.3, .1, .5, .3}, &dev);
  Parameter pWfx("Wfx", {2, 2}, {.4, .1, .5, .8}, &dev);
  Parameter pWox("Wox", {2, 2}, {.5, .9, .9, .7}, &dev);
  Parameter pWjx("Wjx", {2, 2}, {.2, .6, .9, .3}, &dev);
  Parameter pWih("Wih", {2, 2}, {.2, .3, .3, .3}, &dev);
  Parameter pWfh("Wfh", {2, 2}, {.8, .4, .8, .3}, &dev);
  Parameter pWoh("Woh", {2, 2}, {.6, .2, .2, .7}, &dev);
  Parameter pWjh("Wjh", {2, 2}, {.6, .4, .9, .5}, &dev);
  Parameter pbi("bi", {2}, initializers::Constant(0), &dev);
  Parameter pbf("bf", {2}, initializers::Constant(0), &dev);
  Parameter pbo("bo", {2}, initializers::Constant(0), &dev);
  Parameter pbj("bj", {2}, initializers::Constant(0), &dev);

  Graph g;
  using node_ops::matmul;
  using node_ops::input;
  using node_ops::sigmoid;
  using node_ops::tanh;
  using node_ops::zeros;

  const Node x = input(Shape({2}, 2), {2, -2, 0.5, -0.5}, &dev, &g);
  const Node h = input(Shape({2}, 2), {-1, 1, -0.5, 0.5}, &dev, &g);
  const Node c = zeros({2}, &dev, &g);
  const Node Wix = input(&pWix, &g);
  const Node Wfx = input(&pWfx, &g);
  const Node Wox = input(&pWox, &g);
  const Node Wjx = input(&pWjx, &g);
  const Node Wih = input(&pWih, &g);
  const Node Wfh = input(&pWfh, &g);
  const Node Woh = input(&pWoh, &g);
  const Node Wjh = input(&pWjh, &g);
  const Node bi = input(&pbi, &g);
  const Node bf = input(&pbf, &g);
  const Node bo = input(&pbo, &g);
  const Node bj = input(&pbj, &g);

  const Node i = sigmoid(matmul(Wix, x) + matmul(Wih, h) + bi);
  const Node f = sigmoid(matmul(Wfx, x) + matmul(Wfh, h) + bf);
  const Node o = sigmoid(matmul(Wox, x) + matmul(Woh, h) + bo);
  const Node j = tanh(matmul(Wjx, x) + matmul(Wjh, h) + bj);
  const Node cc = f * c + i * j;
  const Node hh = o * tanh(cc);

  const Node t = zeros({2}, &dev, &g);
  const Node diff = hh - t;
  const Node loss = diff * diff;

  EXPECT_EQ(43u, g.num_functions());

  g.forward(loss);
  g.backward(loss);

  const vector<float> expected_losses {
    5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03
  };
  EXPECT_TRUE(vector_match(expected_losses, loss.value().to_vector()));

  auto print = [](const std::string &name, const Tensor &value) {
    std::cout << name << ": shape=" << value.shape().to_string()
      << ", values=[";
    const vector<float> data = value.to_vector();
    for (unsigned i = 0; i < data.size(); ++i) {
      if (i > 0) std::cout << ',';
      std::cout << data[i];
    }
    std::cout << ']' << std::endl;
  };

  std::cout << "VALUES:" << std::endl;
#define PRINT_VALUE(node) print(#node, g.get_value(node))
  PRINT_VALUE(x); PRINT_VALUE(h); PRINT_VALUE(c);
  PRINT_VALUE(Wix); PRINT_VALUE(Wfx); PRINT_VALUE(Wox); PRINT_VALUE(Wjx);
  PRINT_VALUE(Wih); PRINT_VALUE(Wfh); PRINT_VALUE(Woh); PRINT_VALUE(Wjh);
  PRINT_VALUE(bi); PRINT_VALUE(bf); PRINT_VALUE(bo); PRINT_VALUE(bj);
  PRINT_VALUE(i); PRINT_VALUE(f); PRINT_VALUE(o); PRINT_VALUE(j);
  PRINT_VALUE(cc); PRINT_VALUE(hh);
  PRINT_VALUE(t); PRINT_VALUE(diff); PRINT_VALUE(loss);
#undef PRINT_VALUE

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
}

TEST_F(GraphTest, CheckConcatLSTM) {
  // Another implementation of LSTM that concatenates all gates and inputs.
  // All values and gradients should be same as that of "CheckLSTM".
  Parameter pWx("Wx", {8, 2}, {
      .3, .1, .4, .1, .5, .9, .2, .6,
      .5, .3, .5, .8, .9, .7, .9, .3}, &dev);
  Parameter pWh("Wh", {8, 2}, {
      .2, .3, .8, .4, .6, .2, .6, .4,
      .3, .3, .8, .3, .2, .7, .9, .5}, &dev);
  Parameter pb("b", {8}, initializers::Constant(0), &dev);

  Graph g;
  using node_ops::matmul;
  using node_ops::input;
  using node_ops::sigmoid;
  using node_ops::slice;
  using node_ops::tanh;
  using node_ops::zeros;

  const Node x = input(Shape({2}, 2), {2, -2, 0.5, -0.5}, &dev, &g);
  const Node h = input(Shape({2}, 2), {-1, 1, -0.5, 0.5}, &dev, &g);
  const Node c = zeros({2}, &dev, &g);
  const Node Wx = input(&pWx, &g);
  const Node Wh = input(&pWh, &g);
  const Node b = input(&pb, &g);

  const Node u = matmul(Wx, x) + matmul(Wh, h) + b;
  const Node i = sigmoid(slice(u, 0, 0, 2));
  const Node f = sigmoid(slice(u, 0, 2, 4));
  const Node o = sigmoid(slice(u, 0, 4, 6));
  const Node j = tanh(slice(u, 0, 6, 8));
  const Node cc = f * c + i * j;
  const Node hh = o * tanh(cc);

  const Node t = zeros({2}, &dev, &g);
  const Node diff = hh - t;
  const Node loss = diff * diff;

  EXPECT_EQ(26u, g.num_functions());

  g.forward(loss);
  g.backward(loss);

  const vector<float> expected_losses {
    5.7667205e-03, 2.8605087e-02, 1.4819370e-03, 3.0073307e-03
  };
  EXPECT_TRUE(vector_match(expected_losses, loss.value().to_vector()));

  auto print = [](const std::string &name, const Tensor &value) {
    std::cout << name << ": shape=" << value.shape().to_string()
      << ", values=[";
    const vector<float> data = value.to_vector();
    for (unsigned i = 0; i < data.size(); ++i) {
      if (i > 0) std::cout << ',';
      std::cout << data[i];
    }
    std::cout << ']' << std::endl;
  };

  std::cout << "VALUES:" << std::endl;
#define PRINT_VALUE(node) print(#node, g.get_value(node))
  PRINT_VALUE(x); PRINT_VALUE(h); PRINT_VALUE(c);
  PRINT_VALUE(Wx); PRINT_VALUE(Wh); PRINT_VALUE(b);
  PRINT_VALUE(i); PRINT_VALUE(f); PRINT_VALUE(o); PRINT_VALUE(j);
  PRINT_VALUE(cc); PRINT_VALUE(hh);
  PRINT_VALUE(t); PRINT_VALUE(diff); PRINT_VALUE(loss);
#undef PRINT_VALUE

  std::cout << "GRADIENTS:" << std::endl;
#define PRINT_GRAD(node) print(#node, g.get_gradient(node))
  PRINT_GRAD(x); PRINT_GRAD(h); PRINT_GRAD(c);
  PRINT_GRAD(Wx); PRINT_GRAD(Wh); PRINT_GRAD(b);
  PRINT_GRAD(i); PRINT_GRAD(f); PRINT_GRAD(o); PRINT_GRAD(j);
  PRINT_GRAD(cc); PRINT_GRAD(hh);
  PRINT_GRAD(t); PRINT_GRAD(diff); PRINT_GRAD(loss);
#undef PRINT_GRAD
}

}  // namespace primitiv
