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


# SRU cell.
# Formulation:
#   j[t] = W_xj . x[t]
#   f[t] = sigmoid(W_xf . x[t] + b_f)
#   r[t] = sigmoid(W_xr . x[t] + b_r)
#   c[t] = f[t] * c[t-1] + (1 - f[t]) * j[t]
#   h[t] = r[t] * tanh(c[t]) + (1 - r[t]) * x[t]
class SRU(Model):

    def __init__(self, in_size, out_size):
        self.out_size = out_size
        self.pw = Parameter([3 * out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pbf = Parameter([out_size], I.Constant(0))
        self.pbr = Parameter([out_size], I.Constant(0))
        self.add_all_parameters()

    # Initializes internal values.
    def restart(self):
        self.w = F.parameter(self.pw)
        self.bf = F.parameter(self.pbf)
        self.br = F.parameter(self.pbr)

    # Forward.
    def forward(self, xs):
        x = F.concat(xs, 1)
        u = self.w @ x
        j = F.slice(u, 0, 0, self.out_size)
        f = F.sigmoid(
                F.slice(u, 0, self.out_size, 2 * self.out_size)
                + F.broadcast(self.bf, 1, len(xs)))
        r = F.sigmoid(
                F.slice(u, 0, 2 * self.out_size, 3 * self.out_size)
                + F.broadcast(self.bf, 1, len(xs)))
        c = F.zeros([self.out_size])
        hs = []
        for i in range(len(xs)):
            ji = F.slice(j, 1, i, i + 1)
            fi = F.slice(f, 1, i, i + 1)
            ri = F.slice(r, 1, i, i + 1)
            c = fi * c + (1 - fi) * ji
            hs.append(ri * F.tanh(c) + (1 - ri) * xs[i])

        return hs


# Language model using above SRU.
class RNNLM(Model):

    def __init__(self, vocab_size, eos_id):
        self.eos_id = eos_id
        self.plookup = Parameter([NUM_HIDDEN_UNITS, vocab_size], I.Uniform(-0.1, 0.1))
        self.rnn1 = SRU(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS)
        self.rnn2 = SRU(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS)
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

        xs = [F.dropout(F.pick(lookup, inputs[i], 1), DROPOUT_RATE, train)
            for i in range(len(inputs) - 1)]
        hs1 = self.rnn1.forward(xs)
        for i in range(len(inputs) - 1):
            hs1[i] = F.dropout(hs1[i], DROPOUT_RATE, train)
        hs2 = self.rnn2.forward(hs1)
        outputs = [self.hy.forward(F.dropout(hs2[i], DROPOUT_RATE, train))
            for i in range(len(inputs) - 1)]

        return outputs

    # Loss function.
    def loss(self, outputs, inputs):
        losses = [F.softmax_cross_entropy(outputs[i], inputs[i + 1], 0)
            for i in range(len(outputs))]
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
