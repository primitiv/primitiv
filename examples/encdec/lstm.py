# Hand-written LSTM with input/forget/output gates and no peepholes.
# Formulation:
#   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
#   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
#   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
#   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
#   c[t] = i * j + f * c[t-1]
#   h[t] = o * tanh(c[t])

from primitiv import Node
from primitiv import Parameter
from primitiv import operators as F
from primitiv import initializers as I


class LSTM(object):
    def __init__(self, name, in_size, out_size):
        self.name_ = name
        self.out_size_ = out_size
        self.pwxh_ = Parameter([4 * out_size, in_size], I.XavierUniform())
        self.pwhh_ = Parameter([4 * out_size, out_size], I.XavierUniform())
        self.pbh_ = Parameter([4 * out_size], I.Constant(0))

    # Loads all parameters.
    @staticmethod
    def load(name, prefix):
        lstm = LSTM.__new__(LSTM)
        lstm.name_ = name
        lstm.pwxh_ = Parameter.load(prefix + name + "_wxh.param")
        lstm.pwhh_ = Parameter.load(prefix + name + "_whh.param")
        lstm.pbh_ = Parameter.load(prefix + name + "_bh.param")
        lstm.out_size_ = lstm.pbh_.shape()[0] / 4
        return lstm

    # Saves all parameters.
    def save(self, prefix):
        self.pwxh_.save(prefix + self.name_ + "_wxh.param")
        self.pwhh_.save(prefix + self.name_ + "_whh.param")
        self.pbh_.save(prefix + self.name_ + "_bh.param")

    # Adds parameters to the trainer.
    def register_training(self, trainer):
        trainer.add_parameter(self.pwxh_)
        trainer.add_parameter(self.pwhh_)
        trainer.add_parameter(self.pbh_)

    # Initializes internal values.
    def init(self, init_c = Node(), init_h = Node()):
        self.wxh_ = F.parameter(self.pwxh_)
        self.whh_ = F.parameter(self.pwhh_)
        self.bh_ = F.parameter(self.pbh_)
        self.c_ = init_c if init_c.valid() else F.zeros([self.out_size_])
        self.h_ = init_h if init_h.valid() else F.zeros([self.out_size_])

    # One step forwarding.
    def forward(self, x):
        u = self.wxh_ @ x + self.whh_ @ self.h_ + self.bh_
        i = F.sigmoid(F.slice(u, 0, 0, self.out_size_))
        f = F.sigmoid(F.slice(u, 0, self.out_size_, 2 * self.out_size_));
        o = F.sigmoid(F.slice(u, 0, 2 * self.out_size_, 3 * self.out_size_));
        j = F.tanh(F.slice(u, 0, 3 * self.out_size_, 4 * self.out_size_));
        self.c_ = i * j + f * self.c_;
        self.h_ = o * F.tanh(self.c_);
        return self.h_;

    # Retrieves current states.
    def get_c(self):
        return self.c_

    def get_h(self):
        return self.h_
