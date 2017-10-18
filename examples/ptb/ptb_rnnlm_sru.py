#!/usr/bin/env python3

import random
import sys
import math

from primitiv import initializers as I
from primitiv import operators as F
from primitiv import devices as D
from primitiv import trainers as T
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

    def __init__(self, name, in_size, out_size, trainer):
        self.pw_ = Parameter(name + ".w", [out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pb_ = Parameter(name + ".b", [out_size], I.Constant(0))
        trainer.add_parameter(self.pw_)
        trainer.add_parameter(self.pb_)

    # Initializes internal values.
    def init(self):
        self.w_ = F.parameter(self.pw_)
        self.b_ = F.parameter(self.pb_)

    # Applies transform.
    def forward(self, x):
        return self.w_ @ x + self.b_


# SRU cell.
# Formulation:
#   j[t] = W_xj . x[t]
#   f[t] = sigmoid(W_xf . x[t] + b_f)
#   r[t] = sigmoid(W_xr . x[t] + b_r)
#   c[t] = f[t] * c[t-1] + (1 - f[t]) * j[t]
#   h[t] = r[t] * tanh(c[t]) + (1 - r[t]) * x[t]
class SRU(object):

    def __init__(self, name, in_size, out_size, trainer):
        self.out_size_ = out_size
        self.pw_ = Parameter(name + ".w", [3 * out_size, in_size], I.Uniform(-0.1, 0.1))
        self.pbf_ = Parameter(name + ".bf", [out_size], I.Constant(0))
        self.pbr_ = Parameter(name + ".br", [out_size], I.Constant(0))
        trainer.add_parameter(self.pw_)
        trainer.add_parameter(self.pbf_)
        trainer.add_parameter(self.pbr_)

    # Initializes internal values.
    def init(self):
        self.w_ = F.parameter(self.pw_)
        self.bf_ = F.parameter(self.pbf_)
        self.br_ = F.parameter(self.pbr_)

    # Forward.
    def forward(self, xs):
        x = F.concat(xs, 1)
        u = self.w_ @ x
        j = F.slice(u, 0, 0, self.out_size_)
        f = F.sigmoid(
                F.slice(u, 0, self.out_size_, 2 * self.out_size_)
                + F.broadcast(self.bf_, 1, len(xs)))
        r = F.sigmoid(
                F.slice(u, 0, 2 * self.out_size_, 3 * self.out_size_)
                + F.broadcast(self.bf_, 1, len(xs)))
        c = F.zeros([self.out_size_])
        hs = []
        for i in range(len(xs)):
            ji = F.slice(j, 1, i, i + 1)
            fi = F.slice(f, 1, i, i + 1)
            ri = F.slice(r, 1, i, i + 1)
            c = fi * c + (1 - fi) * ji
            hs.append(ri * F.tanh(c) + (1 - ri) * xs[i])

        return hs


# Language model using above SRU.
class RNNLM(object):

    def __init__(self, vocab_size, eos_id, trainer):
        self.eos_id_ = eos_id
        self.plookup_ = Parameter("lookup", [NUM_HIDDEN_UNITS, vocab_size], I.Uniform(-0.1, 0.1))
        self.rnn1_ = SRU("rnn1", NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, trainer)
        self.rnn2_ = SRU("rnn2", NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, trainer)
        self.hy_ = Affine("hy", NUM_HIDDEN_UNITS, vocab_size, trainer)
        trainer.add_parameter(self.plookup_)


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

        xs = [F.dropout(F.pick(lookup, inputs[i], 1), DROPOUT_RATE, train) for i in range(len(inputs) - 1)]
        hs1 = self.rnn1_.forward(xs)
        for i in range(len(inputs) - 1):
            hs1[i] = F.dropout(hs1[i], DROPOUT_RATE, train)
        hs2 = self.rnn2_.forward(hs1)
        outputs = [self.hy_.forward(F.dropout(hs2[i], DROPOUT_RATE, train)) for i in range(len(inputs) - 1)]

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

    # Trainer.
    trainer = T.SGD(1)
    #trainer.set_weight_decay(1e-6)
    trainer.set_gradient_clipping(5)

    # Our LM.
    lm = RNNLM(len(vocab), eos_id, trainer)

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

            trainer.reset_gradients()
            loss.backward()
            trainer.update()

            print("\r%d" % ofs, end="")
            sys.stdout.flush()

        print()

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
            print("\r%d" % ofs, end="")
            sys.stdout.flush()

        print()

        valid_ppl = math.exp(valid_loss / num_valid_labels)
        print("  valid ppl =", valid_ppl)

        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  BEST")
        else:
            old_lr = trainer.get_learning_rate_scaling()
            new_lr = 0.5 * old_lr
            trainer.set_learning_rate_scaling(new_lr)
            print("  learning rate scaled:", old_lr, "->", new_lr)


if __name__ == "__main__":
    main()
