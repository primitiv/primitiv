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
  primitiv::Parameter pwxh_, pwhh_, pbh_;
  Var wxh_, whh_, bh_, h_, c_;

public:
  LSTM() {
    add("wxh", pwxh_);
    add("whh", pwhh_);
    add("bh", pbh_);
  }

  // Initializes the model.
  void init(unsigned in_size, unsigned out_size) {
    using primitiv::initializers::XavierUniform;
    using primitiv::initializers::Constant;
    pwxh_.init({4 * out_size, in_size}, XavierUniform());
    pwhh_.init({4 * out_size, out_size}, XavierUniform());
    pbh_.init({4 * out_size}, Constant(0));
  }

  // Initializes internal values.
  void restart(const Var &init_c = Var(), const Var &init_h = Var()) {
    namespace F = primitiv::functions;
    const unsigned out_size = pwhh_.shape()[1];
    wxh_ = F::parameter<Var>(pwxh_);
    whh_ = F::parameter<Var>(pwhh_);
    bh_ = F::parameter<Var>(pbh_);
    c_ = init_c.valid() ? init_c : F::zeros<Var>({out_size});
    h_ = init_h.valid() ? init_h : F::zeros<Var>({out_size});
  }

  // One step forwarding.
  Var forward(const Var &x) {
    namespace F = primitiv::functions;
    const unsigned out_size = pwhh_.shape()[1];
    const auto u = F::matmul(wxh_, x) + F::matmul(whh_, h_) + bh_;
    const auto i = F::sigmoid(F::slice(u, 0, 0, out_size));
    const auto f = F::sigmoid(F::slice(u, 0, out_size, 2 * out_size));
    const auto o = F::sigmoid(F::slice(u, 0, 2 * out_size, 3 * out_size));
    const auto j = F::tanh(F::slice(u, 0, 3 * out_size, 4 * out_size));
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return h_;
  }

  // Retrieves current states.
  Var get_c() const { return c_; }
  Var get_h() const { return h_; }
};

#endif  // PRIMITIV_EXAMPLE_ENCDEC_LSTM_H_
