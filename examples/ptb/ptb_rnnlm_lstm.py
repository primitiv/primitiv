#!/usr/bin/env python3

import random
import sys
import math

from primitiv import initializers as I
from primitiv import operators as F
from primitiv import devices as D
from primitiv import optimizers as O
from primitiv import Device, Graph, Parameter, Shape

NUM_HIDDEN_UNITS = 650
BATCH_SIZE = 20
MAX_EPOCH = 50
DROPOUT_RATE = 0.5


# Gathers the set of words from space-separated corpus.
def make_vocab(filename):
    vocab = {}
    with open(filename, "r") as ifs:
        for line in ifs:
            line = "<bos> " + line.strip() + " <eos>"
            for word in line.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab


# Generates word ID list using corpus and vocab.
def load_corpus(filename, vocab):
    corpus = []
    with open(filename, "r") as ifs:
        for line in ifs:
            line = "<bos> " + line.strip() + " <eos>"
            sentence = [vocab[word] for word in line.split()]
            corpus.append(sentence)
    return corpus


# Counts output labels in the corpus.
def count_labels(corpus):
    ret = 0
    for sent in corpus:
        ret += len(sent) - 1
    return ret


# Extracts a minibatch from loaded corpus
def make_batch(corpus, sent_ids, eos_id):
    batch_size = len(sent_ids)
    max_len = 0
    for sid in sent_ids:
        max_len = max(max_len, len(corpus[sid]))
    batch = [[eos_id] * batch_size for i in range(max_len)]

    for i in range(batch_size):
        sent = corpus[sent_ids[i]]
        for j in range(len(sent)):
            batch[j][i] = sent[j]

    return batch


# Affine transform:
#   y = W . x + b
class Affine(object):

    def __init__(self, in_size, out_size, optimizer):
        self.pw_ = Parameter([out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pb_ = Parameter([out_size], I.Constant(0))
        optimizer.add_parameter(self.pw_)
        optimizer.add_parameter(self.pb_)

    # Initializes internal values.
    def init(self):
        self.w_ = F.parameter(self.pw_)
        self.b_ = F.parameter(self.pb_)

    # Applies transform.
    def forward(self, x):
        return self.w_ @ x + self.b_


# LSTM with input/forget/output gates and no peepholes.
# Formulation:
#   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
#   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
#   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
#   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
#   c[t] = i * j + f * c[t-1]
#   h[t] = o * tanh(c[t])
class LSTM(object):

    def __init__(self, in_size, out_size, optimizer):
        self.out_size_ = out_size
        self.pwxh_ = Parameter([4 * out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pwhh_ = Parameter([4 * out_size, out_size], I.Uniform(-0.1, 0.1))
        self.pbh_ = Parameter([4 * out_size], I.Constant(0))
        optimizer.add_parameter(self.pwxh_)
        optimizer.add_parameter(self.pwhh_)
        optimizer.add_parameter(self.pbh_)

    # Initializes internal values.
    def init(self):
        self.wxh_ = F.parameter(self.pwxh_)
        self.whh_ = F.parameter(self.pwhh_)
        self.bh_ = F.parameter(self.pbh_)
        self.h_ = self.c_ = F.zeros([self.out_size_])

    # Forward one step.
    def forward(self, x):
        u = self.wxh_ @ x + self.whh_ @ self.h_ + self.bh_
        i = F.sigmoid(F.slice(u, 0, 0, self.out_size_))
        f = F.sigmoid(F.slice(u, 0, self.out_size_, 2 * self.out_size_))
        o = F.sigmoid(F.slice(u, 0, 2 * self.out_size_, 3 * self.out_size_))
        j = F.tanh(F.slice(u, 0, 3 * self.out_size_, 4 * self.out_size_))
        self.c_ = i * j + f * self.c_
        self.h_ = o * F.tanh(self.c_)
        return self.h_


# Language model using above LSTM.
class RNNLM(object):

    def __init__(self, vocab_size, eos_id, optimizer):
        self.eos_id_ = eos_id
        self.plookup_ = Parameter([NUM_HIDDEN_UNITS, vocab_size], I.Uniform(-0.1, 0.1))
        self.rnn1_ = LSTM(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, optimizer)
        self.rnn2_ = LSTM(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, optimizer)
        self.hy_ = Affine(NUM_HIDDEN_UNITS, vocab_size, optimizer)
        optimizer.add_parameter(self.plookup_)


    # Forward function of RNNLM. Input data should be arranged below:
    # inputs = {
    #   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<bos>)
    #   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
    #   ...,
    #   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<eos>)
    # };
    def forward(self, inputs, train):
        batch_size = len(inputs[0])
        lookup = F.parameter(self.plookup_)
        self.rnn1_.init()
        self.rnn2_.init()
        self.hy_.init()

        outputs = []
        for i in range(len(inputs) - 1):
            x = F.pick(lookup, inputs[i], 1)
            x = F.dropout(x, DROPOUT_RATE, train)
            h1 = self.rnn1_.forward(x)
            h1 = F.dropout(h1, DROPOUT_RATE, train)
            h2 = self.rnn2_.forward(h1)
            h2 = F.dropout(h2, DROPOUT_RATE, train)
            outputs.append(self.hy_.forward(h2))

        return outputs


    # Loss function.
    def loss(self, outputs, inputs):
        losses = [F.softmax_cross_entropy(outputs[i], inputs[i + 1], 0) for i in range(len(outputs))]
        return F.batch.mean(F.sum(losses))


def main():
    # Loads vocab.
    vocab = make_vocab("data/ptb.train.txt")
    print("#vocab:", len(vocab))  # maybe 10001
    eos_id = vocab["<eos>"]

    # Loads all corpus.
    train_corpus = load_corpus("data/ptb.train.txt", vocab)
    valid_corpus = load_corpus("data/ptb.valid.txt", vocab)
    num_train_sents = len(train_corpus)
    num_valid_sents = len(valid_corpus)
    num_train_labels = count_labels(train_corpus)
    num_valid_labels = count_labels(valid_corpus)
    print("train:", num_train_sents, "sentences,", num_train_labels, "labels")
    print("valid:", num_valid_sents, "sentences,", num_valid_labels, "labels")

    dev = D.CUDA(0)
    Device.set_default(dev)

    # Optimizer.
    optimizer = O.SGD(1)
    #optimizer.set_weight_decay(1e-6)
    optimizer.set_gradient_clipping(5)

    # Our LM.
    lm = RNNLM(len(vocab), eos_id, optimizer)

    # Sentence IDs.
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    best_valid_ppl = 1e10

    g = Graph()
    Graph.set_default(g)

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
