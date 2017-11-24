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

NUM_HIDDEN_UNITS = 256
BATCH_SIZE = 64
MAX_EPOCH = 100


class RNNLM(Model):

    def __init__(self, vocab_size, eos_id):
        self.eos_id = eos_id
        self.pwlookup = Parameter([NUM_HIDDEN_UNITS, vocab_size], I.XavierUniform())
        self.pwxs = Parameter([NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS], I.XavierUniform())
        self.pwsy = Parameter([vocab_size, NUM_HIDDEN_UNITS], I.XavierUniform())
        self.add_all_parameters()

    # Forward function of RNNLM. Input data should be arranged below:
    # inputs = {
    #   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<s>)
    #   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
    #
    #   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<s>)
    # };
    def forward(self, inputs):
        batch_size = len(inputs[0])
        wlookup = F.parameter(self.pwlookup)
        wxs = F.parameter(self.pwxs)
        wsy = F.parameter(self.pwsy)
        s = F.zeros(Shape([NUM_HIDDEN_UNITS], batch_size))
        outputs = []
        for i in range(len(inputs) - 1):
            w = F.pick(wlookup, inputs[i], 1)
            x = w + s
            s = F.sigmoid(wxs @ x)
            outputs.append(wsy @ s)
        return outputs

    # Loss function.
    def forward_loss(self, outputs, inputs):
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
    dev = D.CUDA(0);
    Device.set_default(dev)
    g = Graph();
    Graph.set_default(g)

    # Our LM.
    lm = RNNLM(len(vocab), eos_id)

    # Optimizer.
    optimizer = O.Adam()
    optimizer.set_weight_decay(1e-6)
    optimizer.set_gradient_clipping(5)
    optimizer.add_model(lm)

    # Sentence IDs.
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    # Train/valid loop.
    for epoch in range(MAX_EPOCH):
        print("epoch", (epoch + 1), "/", MAX_EPOCH, ":")
        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            batch_ids = train_ids[ofs : min(ofs + BATCH_SIZE, num_train_sents)]
            batch = make_batch(train_corpus, batch_ids, eos_id)

            g.clear()

            outputs = lm.forward(batch)
            loss = lm.forward_loss(outputs, batch)
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

            outputs = lm.forward(batch)
            loss = lm.forward_loss(outputs, batch)
            valid_loss += loss.to_float() * len(batch_ids)
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

        valid_ppl = math.exp(valid_loss / num_valid_labels)
        print("  valid ppl =", valid_ppl)


if __name__ == "__main__":
    main()
