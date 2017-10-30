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
class LSTM {
  std::string name_;
  unsigned out_size_;
  primitiv::Parameter pwxh_, pwhh_, pbh_;
  Var wxh_, whh_, bh_, h_, c_;

public:
  LSTM(const std::string &name, unsigned in_size, unsigned out_size)
    : name_(name)
    , out_size_(out_size)
    , pwxh_({4 * out_size, in_size}, primitiv::initializers::XavierUniform())
    , pwhh_({4 * out_size, out_size}, primitiv::initializers::XavierUniform())
    , pbh_({4 * out_size}, primitiv::initializers::Constant(0)) {}

  // Loads all parameters.
  LSTM(const std::string &name, const std::string &prefix)
    : name_(name)
    , pwxh_(primitiv::Parameter::load(prefix + name_ + "_wxh.param"))
    , pwhh_(primitiv::Parameter::load(prefix + name_ + "_whh.param"))
    , pbh_(primitiv::Parameter::load(prefix + name_ + "_bh.param")) {
      out_size_ = pbh_.shape()[0] / 4;
  }

  // Saves all parameters.
  void save(const std::string &prefix) const {
    pwxh_.save(prefix + name_ + "_wxh.param");
    pwhh_.save(prefix + name_ + "_whh.param");
    pbh_.save(prefix + name_ + "_bh.param");
  }

  // Adds parameters to the trainer.
  void register_training(primitiv::Trainer &trainer) {
    trainer.add_parameter(pwxh_);
    trainer.add_parameter(pwhh_);
    trainer.add_parameter(pbh_);
  }

  // Initializes internal values.
  void init(
      const Var &init_c = Var(),
      const Var &init_h = Var()) {
    namespace F = primitiv::operators;
    wxh_ = F::parameter<Var>(pwxh_);
    whh_ = F::parameter<Var>(pwhh_);
    bh_ = F::parameter<Var>(pbh_);
    c_ = init_c.valid() ? init_c : F::zeros<Var>({out_size_});
    h_ = init_h.valid() ? init_h : F::zeros<Var>({out_size_});
  }

  // One step forwarding.
  Var forward(const Var &x) {
    namespace F = primitiv::operators;
    const auto u = F::matmul(wxh_, x) + F::matmul(whh_, h_) + bh_;
    const auto i = F::sigmoid(F::slice(u, 0, 0, out_size_));
    const auto f = F::sigmoid(F::slice(u, 0, out_size_, 2 * out_size_));
    const auto o = F::sigmoid(F::slice(u, 0, 2 * out_size_, 3 * out_size_));
    const auto j = F::tanh(F::slice(u, 0, 3 * out_size_, 4 * out_size_));
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return h_;
  }

  // Retrieves current states.
  Var get_c() const { return c_; }
  Var get_h() const { return h_; }
};

#endif  // PRIMITIV_EXAMPLE_ENCDEC_LSTM_H_
