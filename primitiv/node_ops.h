#ifndef PRIMITIV_NODE_OPS_H_
#define PRIMITIV_NODE_OPS_H_

#include <vector>
#include <primitiv/function_impl.h>
#include <primitiv/graph.h>
#include <primitiv/node.h>
#include <primitiv/shape.h>
#include <primitiv/parameter.h>

#define APP(x) (x).graph().add_function
#define F functions

namespace primitiv {

inline Node operator+(const Node &x) { return APP(x)(new F::Positive(), {x}); }
inline Node operator-(const Node &x) { return APP(x)(new F::Negative(), {x}); }
inline Node operator+(const Node &x, const float k) { return APP(x)(new F::AddConst(k), {x}); }
inline Node operator+(const float k, const Node &x) { return APP(x)(new F::AddConst(k), {x}); }
inline Node operator+(const Node &a, const Node &b) { return APP(a)(new F::Add(), {a, b}); }
inline Node operator-(const Node &x, const float k) { return APP(x)(new F::SubtractConstR(k), {x}); }
inline Node operator-(const float k, const Node &x) { return APP(x)(new F::SubtractConstL(k), {x}); }
inline Node operator-(const Node &a, const Node &b) { return APP(a)(new F::Subtract(), {a, b}); }
inline Node operator*(const Node &x, const float k) { return APP(x)(new F::MultiplyConst(k), {x}); }
inline Node operator*(const float k, const Node &x) { return APP(x)(new F::MultiplyConst(k), {x}); }
inline Node operator*(const Node &a, const Node &b) { return APP(a)(new F::Multiply(), {a, b}); }
inline Node operator/(const Node &x, const float k) { return APP(x)(new F::DivideConstR(k), {x}); }
inline Node operator/(const float k, const Node &x) { return APP(x)(new F::DivideConstL(k), {x}); }
inline Node operator/(const Node &a, const Node &b) { return APP(a)(new F::Divide(), {a, b}); }

namespace node_ops {

inline Node input(
    Graph &g,
    Device &dev,
    const Shape &shape,
    const std::vector<float> &data) {
  return g.add_function(new F::Input(shape, &dev, data), {});
}

inline Node parameter(Graph &g, Parameter &param) {
  return g.add_function(new F::ParameterInput(param), {});
}

inline Node transpose(const Node &x) { return APP(x)(new F::Transpose(), {x}); }
inline Node dot(const Node &a, const Node &b) { return APP(a)(new F::Dot(), {a, b}); }
inline Node exp(const Node &x) { return APP(x)(new F::Exp(), {x}); }
inline Node tanh(const Node &x) { return APP(x)(new F::Tanh(), {x}); }

}  // namespace node_ops
}  // namespace primitiv

#undef APP
#undef F

#endif  // PRIMITIV_NODE_OPS_H_
