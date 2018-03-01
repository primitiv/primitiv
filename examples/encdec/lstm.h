#ifndef PRIMITIV_EXAMPLE_ENCDEC_LSTM_H_
#define PRIMITIV_EXAMPLE_ENCDEC_LSTM_H_

#include <string>

#include <primitiv/primitiv.h>

// Hand-written LSTM with input/forget/output gates and no peepholes.
// Formulation:
//   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
//   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
//   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
//   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
//   c[t] = i * j + f * c[t-1]
//   h[t] = o * tanh(c[t])
template<typename Var>
class LSTM : public primitiv::Model {
  primitiv::Parameter pw_, pb_;
  Var w_, b_, h_, c_;

public:
  LSTM() {
    add("w", pw_);
    add("b", pb_);
  }

  // Initializes the model.
  void init(unsigned in_size, unsigned out_size) {
    using primitiv::initializers::Uniform;
    using primitiv::initializers::Constant;
    pw_.init({4 * out_size, in_size + out_size}, Uniform(-0.1, 0.1));
    pb_.init({4 * out_size}, Constant(0));
  }

  // Initializes internal values.
  void restart(const Var &init_c = Var(), const Var &init_h = Var()) {
    namespace F = primitiv::functions;
    const unsigned out_size = pb_.shape()[0] / 4;
    w_ = F::parameter<Var>(pw_);
    b_ = F::parameter<Var>(pb_);
    c_ = init_c.valid() ? init_c : F::zeros<Var>({out_size});
    h_ = init_h.valid() ? init_h : F::zeros<Var>({out_size});
  }

  // One step forwarding.
  Var forward(const Var &x) {
    namespace F = primitiv::functions;
    const auto u = F::matmul(w_, F::concat({x, h_}, 0)) + b_;
    const std::vector<Var> v = F::split(u, 0, 4);
    const auto i = F::sigmoid(v[0]);
    const auto f = F::sigmoid(v[1]);
    const auto o = F::sigmoid(v[2]);
    const auto j = F::tanh(v[3]);
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return h_;
  }

  // Retrieves current states.
  Var get_c() const { return c_; }
  Var get_h() const { return h_; }
};

#endif  // PRIMITIV_EXAMPLE_ENCDEC_LSTM_H_
