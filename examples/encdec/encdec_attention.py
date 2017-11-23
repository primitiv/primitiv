#!/usr/bin/env python3
# coding: utf-8

import sys
import random
import math

import numpy as np

from primitiv import Device, Graph, Model, Parameter, Optimizer

from primitiv import devices as D
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import optimizers as O

from lstm import LSTM
from utils import (
    make_vocab, load_corpus, load_corpus_ref, count_labels, make_batch,
    save_ppl, make_inv_vocab, line_to_sent, load_ppl
)

from argparse import ArgumentParser
from bleu import get_bleu_stats, calculate_bleu
from collections import defaultdict

SRC_VOCAB_SIZE = 4000
TRG_VOCAB_SIZE = 5000
NUM_EMBED_UNITS = 512
NUM_HIDDEN_UNITS = 512
BATCH_SIZE = 64
MAX_EPOCH = 100
GENERATION_LIMIT = 32
DROPOUT_RATE = 0.5

SRC_TRAIN_FILE = "data/train.en"
TRG_TRAIN_FILE = "data/train.ja"
SRC_VALID_FILE = "data/dev.en"
TRG_VALID_FILE = "data/dev.ja"
SRC_TEST_FILE = "data/test.en"
REF_TEST_FILE = "data/test.ja"

class AttentionalEncoderDecoder(Model):
    """Encoder-decoder translation model with dot-attention."""
    
    def __init__(self):
        self.dropout_rate = DROPOUT_RATE
        self.psrc_lookup = Parameter()
        self.ptrg_lookup = Parameter()
        self.pwhj = Parameter()
        self.pbj = Parameter()
        self.pwjy = Parameter()
        self.pby = Parameter()
        self.src_fw_lstm = LSTM()
        self.src_bw_lstm = LSTM()
        self.trg_lstm = LSTM()
        self.add_all_parameters()
        self.add_all_submodels()

    def init(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size):
        """Creates a new AttentionalEncoderDecoder object."""
        self.psrc_lookup.init([embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup.init([embed_size, trg_vocab_size], I.XavierUniform())
        self.pwhj.init([embed_size, 2 * hidden_size], I.XavierUniform())
        self.pbj.init([embed_size], I.Constant(0))
        self.pwjy.init([trg_vocab_size, embed_size], I.XavierUniform())
        self.pby.init([trg_vocab_size], I.Constant(0))
        self.src_fw_lstm.init(embed_size, hidden_size)
        self.src_bw_lstm.init(embed_size, hidden_size)
        self.trg_lstm.init(2 * embed_size, hidden_size)

    def encode(self, src_batch, train):
        """Encodes source sentences and prepares internal states."""
        # Embedding lookup.
        src_lookup = F.parameter(self.psrc_lookup)
        e_list = []
        for x in src_batch:
            e = F.pick(src_lookup, x, 1)
            e = F.dropout(e, self.dropout_rate, train)
            e_list.append(e)

        # Forward encoding
        self.src_fw_lstm.restart()
        f_list = []
        for e in e_list:
            f = self.src_fw_lstm.forward(e)
            f = F.dropout(f, self.dropout_rate, train)
            f_list.append(f)

        # Backward encoding
        self.src_bw_lstm.restart()
        b_list = []
        for e in reversed(e_list):
            b = self.src_bw_lstm.forward(e)
            b = F.dropout(b, self.dropout_rate, train)
            b_list.append(b)

        b_list.reverse()

        # Concatenates RNN states.
        fb_list = [f_list[i] + b_list[i] for i in range(len(src_batch))]
        self.concat_fb = F.concat(fb_list, 1)
        self.t_concat_fb = F.transpose(self.concat_fb)

        # Initializes decode states.
        embed_size = self.psrc_lookup.shape()[0]
        self.trg_lookup = F.parameter(self.ptrg_lookup)
        self.whj = F.parameter(self.pwhj)
        self.bj = F.parameter(self.pbj)
        self.wjy = F.parameter(self.pwjy)
        self.by = F.parameter(self.pby)
        self.feed = F.zeros([embed_size])
        self.trg_lstm.restart(
            self.src_fw_lstm.get_c() + self.src_bw_lstm.get_c(),
            self.src_fw_lstm.get_h() + self.src_bw_lstm.get_h())

    def decode_step(self, trg_words, train):
        """One step decoding."""
        e = F.pick(self.trg_lookup, trg_words, 1)
        e = F.dropout(e, self.dropout_rate, train)
        h = self.trg_lstm.forward(F.concat([e, self.feed], 0))
        h = F.dropout(h, self.dropout_rate, train)
        atten_probs = F.softmax(self.t_concat_fb @ h, 0)
        c = self.concat_fb @ atten_probs
        self.feed = F.tanh(self.whj @ F.concat([h, c], 0) + self.bj)
        return self.wjy @ self.feed + self.by

    def loss(self, trg_batch, train):
        """Calculates loss values."""
        losses = []
        for i in range(len(trg_batch)-1):
            y = self.decode_step(trg_batch[i], train)
            loss = F.softmax_cross_entropy(y, trg_batch[i + 1], 0)
            losses.append(loss)
        return F.batch.mean(F.sum(losses))


def train(encdec, optimizer, prefix, best_valid_ppl):
    # Registers all parameters to the optimizer.
    optimizer.add_model(encdec)

    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)
    print("#src_vocab:", len(src_vocab))
    print("#trg_vocab:", len(trg_vocab))

    # Loads all corpus
    train_src_corpus = load_corpus(SRC_TRAIN_FILE, src_vocab)
    train_trg_corpus = load_corpus(TRG_TRAIN_FILE, trg_vocab)
    valid_src_corpus = load_corpus(SRC_VALID_FILE, src_vocab)
    valid_trg_corpus = load_corpus(TRG_VALID_FILE, trg_vocab)
    test_src_corpus = load_corpus(SRC_TEST_FILE, src_vocab)
    test_ref_corpus = load_corpus_ref(REF_TEST_FILE, trg_vocab)
    num_train_sents = len(train_trg_corpus)
    num_valid_sents = len(valid_trg_corpus)
    num_test_sents = len(test_ref_corpus)
    num_train_labels = count_labels(train_trg_corpus)
    num_valid_labels = count_labels(valid_trg_corpus)
    print("train:", num_train_sents, "sentences,", num_train_labels, "labels")
    print("valid:", num_valid_sents, "sentences,", num_valid_labels, "labels")

    # Sentence IDs
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    # Train/valid loop.
    for epoch in range(MAX_EPOCH):
        # Computation graph.
        g = Graph()
        Graph.set_default(g)

        print("epoch %d/%d:" % (epoch + 1, MAX_EPOCH))
        print("  learning rate scale = %.4e" % optimizer.get_learning_rate_scaling())

        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0.
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = train_ids[ofs:min(ofs + BATCH_SIZE, num_train_sents)]
            src_batch = make_batch(train_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(train_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, True)
            loss = encdec.loss(trg_batch, True)
            train_loss += loss.to_float() * len(batch_ids)

            optimizer.reset_gradients()
            loss.backward()
            optimizer.update()

        train_ppl = math.exp(train_loss / num_train_labels)
        print("  train PPL = %.4f" % train_ppl)

        # Validation.
        valid_loss = 0.
        for ofs in range(0, num_valid_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = valid_ids[ofs:min(ofs + BATCH_SIZE, num_valid_sents)]
            src_batch = make_batch(valid_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(valid_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, False)
            loss = encdec.loss(trg_batch, False)
            valid_loss += loss.to_float() * len(batch_ids)

        valid_ppl = math.exp(valid_loss/num_valid_labels)
        print("  valid PPL = %.4f" % valid_ppl)

        # Calculates test BLEU.
        stats = defaultdict(int)
        for ofs in range(0, num_test_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            src_batch = test_src_corpus[ofs : min(ofs + BATCH_SIZE, num_test_sents)]
            ref_batch = test_ref_corpus[ofs : min(ofs + BATCH_SIZE, num_test_sents)]

            hyp_ids = test_batch(encdec, src_vocab, trg_vocab, src_batch)
            for hyp_line, ref_line in zip(hyp_ids, ref_batch):
                for k, v in get_bleu_stats(ref_line[1:-1], hyp_line).items():
                    stats[k] += v

        bleu = calculate_bleu(stats)
        print("  test BLEU = %.2f" % (100 * bleu))

        # Saves best model/optimizer.
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  saving model/optimizer ... ", end="")
            sys.stdout.flush()
            encdec.save(prefix + ".model")
            optimizer.save(prefix + ".optimizer")
            save_ppl(prefix + ".valid_ppl", best_valid_ppl)
            print("done.")
        else:
            # Learning rate decay by 1/sqrt(2)
            new_scale = .7071 * optimizer.get_learning_rate_scaling()
            optimizer.set_learning_rate_scaling(new_scale)


def test_batch(encdec, src_vocab, trg_vocab, lines):
    g = Graph()
    Graph.set_default(g)

    src_batch = make_batch(lines, list(range(len(lines))), src_vocab)

    encdec.encode(src_batch, False)

    # Generates target words one-by-one.
    trg_ids = [np.array([trg_vocab["<bos>"]] * len(lines))]
    eos_id = trg_vocab["<eos>"]
    eos_ids = np.array([eos_id] * len(lines))
    while (trg_ids[-1] != eos_ids).any():
        if len(trg_ids) > GENERATION_LIMIT + 1:
            print("Warning: Sentence generation did not finish in",
                    GENERATION_LIMIT, "iterations.", file=sys.stderr)
            trg_ids.append(eos_ids)
            break
        y = encdec.decode_step(trg_ids[-1], False)
        trg_ids.append(np.array(y.argmax(0)).T)

    return [hyp[:np.where(hyp == eos_id)[0][0]] for hyp in np.array(trg_ids[1:]).T]


def test(encdec):
    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)

    for line in sys.stdin:
        trg_ids = test_batch(encdec, src_vocab, trg_vocab, [line_to_sent(line.strip(), src_vocab)])[0]
        # Prints the result.
        print(" ".join(inv_trg_vocab[wid] for wid in trg_ids))


def main():
    parser = ArgumentParser()
    parser.add_argument("mode", help="(train|resume|test)")
    parser.add_argument("model_prefix", help="prefix of the model files.")
    args = parser.parse_args()

    mode = args.mode
    prefix = args.model_prefix
    print("mode:", mode, file=sys.stderr)
    print("prefix:", prefix, file=sys.stderr)

    if mode not in ("train", "resume", "test"):
        print("unknown mode:", mode, file=sys.stderr)
        return

    print("initializing device ... ", end="", file=sys.stderr)
    sys.stderr.flush()
    dev = D.CUDA(0)
    Device.set_default(dev)
    print("done.", file=sys.stderr)

    if mode == "train":
        encdec = AttentionalEncoderDecoder()
        encdec.init(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, NUM_EMBED_UNITS, NUM_HIDDEN_UNITS)
        optimizer = O.Adam()
        optimizer.set_weight_decay(1e-6)
        optimizer.set_gradient_clipping(5)
        train(encdec, optimizer, prefix, 1e10)
    elif mode == "resume":
        print("loading model/optimizer ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = AttentionalEncoderDecoder()
        encdec.load(prefix + ".model")
        optimizer = O.Adam()
        optimizer.load(prefix + ".optimizer")
        valid_ppl = load_ppl(prefix + ".valid_ppl")
        print("done.", file=sys.stderr)
        train(encdec, optimizer, prefix, valid_ppl)
    else:
        print("loading model ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = AttentionalEncoderDecoder()
        encdec.load(prefix + ".model")
        print("done.", file=sys.stderr)
        test(encdec)


if __name__ == "__main__":
    main()
