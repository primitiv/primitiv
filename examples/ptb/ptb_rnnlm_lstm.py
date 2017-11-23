#!/usr/bin/env python3

import random
import sys
import math

from primitiv import initializers as I
from primitiv import operators as F
from primitiv import devices as D
from primitiv import optimizers as O
from primitiv import Device, Graph, Model, Parameter, Shape

from utils import make_vocab, load_corpus, count_labels, make_batch

NUM_HIDDEN_UNITS = 650
BATCH_SIZE = 20
MAX_EPOCH = 50
DROPOUT_RATE = 0.5


# Affine transform:
#   y = W . x + b
class Affine(Model):

    def __init__(self, in_size, out_size):
        self.pw = Parameter([out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pb = Parameter([out_size], I.Constant(0))
        self.add_all_parameters()

    # Initializes internal values.
    def reset(self):
        self.w = F.parameter(self.pw)
        self.b = F.parameter(self.pb)

    # Applies transform.
    def forward(self, x):
        return self.w @ x + self.b


# LSTM with input/forget/output gates and no peepholes.
# Formulation:
#   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
#   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
#   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
#   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
#   c[t] = i * j + f * c[t-1]
#   h[t] = o * tanh(c[t])
class LSTM(Model):

    def __init__(self, in_size, out_size):
        self.out_size = out_size
        self.pwxh = Parameter([4 * out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pwhh = Parameter([4 * out_size, out_size], I.Uniform(-0.1, 0.1))
        self.pbh = Parameter([4 * out_size], I.Constant(0))
        self.add_all_parameters()

    # Initializes internal values.
    def restart(self):
        self.wxh = F.parameter(self.pwxh)
        self.whh = F.parameter(self.pwhh)
        self.bh = F.parameter(self.pbh)
        self.h = self.c = F.zeros([self.out_size])

    # Forward one step.
    def forward(self, x):
        u = self.wxh @ x + self.whh @ self.h + self.bh
        i = F.sigmoid(F.slice(u, 0, 0, self.out_size))
        f = F.sigmoid(F.slice(u, 0, self.out_size, 2 * self.out_size))
        o = F.sigmoid(F.slice(u, 0, 2 * self.out_size, 3 * self.out_size))
        j = F.tanh(F.slice(u, 0, 3 * self.out_size, 4 * self.out_size))
        self.c = i * j + f * self.c
        self.h = o * F.tanh(self.c)
        return self.h


# Language model using above LSTM.
class RNNLM(Model):

    def __init__(self, vocab_size, eos_id):
        self.eos_id = eos_id
        self.plookup = Parameter([NUM_HIDDEN_UNITS, vocab_size], I.Uniform(-0.1, 0.1))
        self.rnn1 = LSTM(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS)
        self.rnn2 = LSTM(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS)
        self.hy = Affine(NUM_HIDDEN_UNITS, vocab_size)
        self.add_all_parameters()
        self.add_all_submodels()


    # Forward function of RNNLM. Input data should be arranged below:
    # inputs = {
    #   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<bos>)
    #   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
    #   ...,
    #   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<eos>)
    # };
    def forward(self, inputs, train):
        batch_size = len(inputs[0])
        lookup = F.parameter(self.plookup)
        self.rnn1.restart()
        self.rnn2.restart()
        self.hy.reset()

        outputs = []
        for i in range(len(inputs) - 1):
            x = F.pick(lookup, inputs[i], 1)
            x = F.dropout(x, DROPOUT_RATE, train)
            h1 = self.rnn1.forward(x)
            h1 = F.dropout(h1, DROPOUT_RATE, train)
            h2 = self.rnn2.forward(h1)
            h2 = F.dropout(h2, DROPOUT_RATE, train)
            outputs.append(self.hy.forward(h2))

        return outputs


    # Loss function.
    def loss(self, outputs, inputs):
        losses = [F.softmax_cross_entropy(outputs[i], inputs[i + 1], 0) for i in range(len(outputs))]
        return F.batch.mean(F.sum(losses))


def main():
    # Loads vocab.
    vocab = make_vocab("data/ptb.train.txt")
    print("#vocab:", len(vocab))  # maybe 10000
    eos_id = vocab["<s>"]

    # Loads all corpus.
    train_corpus = load_corpus("data/ptb.train.txt", vocab)
    valid_corpus = load_corpus("data/ptb.valid.txt", vocab)
    num_train_sents = len(train_corpus)
    num_valid_sents = len(valid_corpus)
    num_train_labels = count_labels(train_corpus)
    num_valid_labels = count_labels(valid_corpus)
    print("train:", num_train_sents, "sentences,", num_train_labels, "labels")
    print("valid:", num_valid_sents, "sentences,", num_valid_labels, "labels")

    # Device and computation graph.
    dev = D.CUDA(0)
    Device.set_default(dev)
    g = Graph()
    Graph.set_default(g)

    # Our LM.
    lm = RNNLM(len(vocab), eos_id)

    # Optimizer.
    optimizer = O.SGD(1)
    #optimizer.set_weight_decay(1e-6)
    optimizer.set_gradient_clipping(5)
    optimizer.add_model(lm)

    # Sentence IDs.
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    best_valid_ppl = 1e10

    # Train/valid loop.
    for epoch in range(MAX_EPOCH):
        print("epoch", epoch + 1, "/", MAX_EPOCH, ":")
        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            batch_ids = train_ids[ofs : min(ofs + BATCH_SIZE, num_train_sents)]
            batch = make_batch(train_corpus, batch_ids, eos_id)

            g.clear()

            outputs = lm.forward(batch, True)
            loss = lm.loss(outputs, batch)
            train_loss += loss.to_float() * len(batch_ids)

            optimizer.reset_gradients()
            loss.backward()
            optimizer.update()

            print("%d" % ofs, end="\r")
            sys.stdout.flush()

        train_ppl = math.exp(train_loss / num_train_labels)
        print("  train ppl =", train_ppl)

        # Validation.
        valid_loss = 0
        for ofs in range(0, num_valid_sents, BATCH_SIZE):
            batch_ids = valid_ids[ofs : min(ofs + BATCH_SIZE, num_valid_sents)]
            batch = make_batch(valid_corpus, batch_ids, eos_id)

            g.clear()

            outputs = lm.forward(batch, False)
            loss = lm.loss(outputs, batch)
            valid_loss += loss.to_float() * len(batch_ids)
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

        valid_ppl = math.exp(valid_loss / num_valid_labels)
        print("  valid ppl =", valid_ppl)

        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  BEST")
        else:
            old_lr = optimizer.get_learning_rate_scaling()
            new_lr = 0.5 * old_lr
            optimizer.set_learning_rate_scaling(new_lr)
            print("  learning rate scaled:", old_lr, "->", new_lr)


if __name__ == "__main__":
    main()
