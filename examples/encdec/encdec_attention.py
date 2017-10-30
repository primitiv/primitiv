#!/usr/bin/env python3
# coding: utf-8

import sys
import random
import math

import numpy as np

from primitiv import Device, Parameter, Graph, Trainer

from primitiv import devices as D
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import trainers as T

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

# Encoder-decoder translation model with dot-attention.
class EncoderDecoder(object):
    def __init__(self, name, src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_rate):
        self.name_ = name
        self.embed_size_ = embed_size
        self.dropout_rate_ = dropout_rate
        self.psrc_lookup_ = Parameter([embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup_ = Parameter([embed_size, trg_vocab_size], I.XavierUniform())
        self.pwhj_ = Parameter([embed_size, 2*hidden_size], I.XavierUniform())
        self.pbj_ = Parameter([embed_size], I.Constant(0))
        self.pwjy_ = Parameter([trg_vocab_size, embed_size], I.XavierUniform())
        self.pby_ = Parameter([trg_vocab_size], I.Constant(0))
        self.src_fw_lstm_ = LSTM(name+"_src_fw_lstm", embed_size, hidden_size)
        self.src_bw_lstm_ = LSTM(name+"_src_bw_lstm", embed_size, hidden_size)
        self.trg_lstm_ = LSTM(name+"_trg_lstm", embed_size*2, hidden_size)

    # Loads all parameters.
    @staticmethod
    def load(name, prefix):
        encdec = EncoderDecoder.__new__(EncoderDecoder)
        encdec.name_ = name
        encdec.psrc_lookup_ = Parameter.load(prefix+name+"_src_lookup.param")
        encdec.ptrg_lookup_ = Parameter.load(prefix+name+"_trg_lookup.param")
        encdec.pwhj_ = Parameter.load(prefix+name+"_whj.param")
        encdec.pbj_ = Parameter.load(prefix+name+"_bj.param")
        encdec.pwjy_ = Parameter.load(prefix+name+"_wjy.param")
        encdec.pby_ = Parameter.load(prefix+name+"_by.param")
        encdec.src_fw_lstm_ = LSTM.load(name+"_src_fw_lstm", prefix)
        encdec.src_bw_lstm_ = LSTM.load(name+"_src_bw_lstm", prefix)
        encdec.trg_lstm_ = LSTM.load(name+"_trg_lstm", prefix)
        encdec.embed_size_ = encdec.pbj_.shape()[0]
        with open(prefix+name+".config", "r", encoding="utf-8") as f:
            encdec.dropout_rate_ = float(f.readline())
        return encdec

    # Saves all parameters
    def save(self, prefix):
        self.psrc_lookup_.save(prefix+self.name_+"_src_lookup.param")
        self.ptrg_lookup_.save(prefix+self.name_+"_trg_lookup.param")
        self.pwhj_.save(prefix+self.name_+"_whj.param")
        self.pbj_.save(prefix+self.name_+"_bj.param")
        self.pwjy_.save(prefix+self.name_+"_wjy.param")
        self.pby_.save(prefix+self.name_+"_by.param")
        self.src_fw_lstm_.save(prefix)
        self.src_bw_lstm_.save(prefix)
        self.trg_lstm_.save(prefix)
        with open(prefix+self.name_+".config", "w", encoding="utf-8") as f:
            print(self.dropout_rate_, file=f)

    # Adds parameters to the trainer
    def register_training(self, trainer):
        trainer.add_parameter(self.psrc_lookup_)
        trainer.add_parameter(self.ptrg_lookup_)
        trainer.add_parameter(self.pwhj_)
        trainer.add_parameter(self.pbj_)
        trainer.add_parameter(self.pwjy_)
        trainer.add_parameter(self.pby_)
        self.src_fw_lstm_.register_training(trainer)
        self.src_bw_lstm_.register_training(trainer)
        self.trg_lstm_.register_training(trainer)

    # Encodes source sentences and prepare internal states.
    def encode(self, src_batch, train):
        # Embedding lookup.
        src_lookup = F.parameter(self.psrc_lookup_)
        e_list = []
        for x in src_batch:
            e = F.pick(src_lookup, x, 1)
            e = F.dropout(e, self.dropout_rate_, train)
            e_list.append(e)

        # Forward encoding
        self.src_fw_lstm_.init()
        f_list = []
        for e in e_list:
            f = self.src_fw_lstm_.forward(e)
            f = F.dropout(f, self.dropout_rate_, train)
            f_list.append(f)

        # Backward encoding
        self.src_bw_lstm_.init()
        b_list = []
        for e in reversed(e_list):
            b = self.src_bw_lstm_.forward(e)
            b = F.dropout(b, self.dropout_rate_, train)
            b_list.append(b)

        b_list.reverse()

        # Concatenates RNN states.
        fb_list = [f_list[i] + b_list[i] for i in range(len(src_batch))]
        self.concat_fb_ = F.concat(fb_list, 1)
        self.t_concat_fb_ = F.transpose(self.concat_fb_)

        # Initializes decode states.
        self.trg_lookup_ = F.parameter(self.ptrg_lookup_)
        self.whj_ = F.parameter(self.pwhj_)
        self.bj_ = F.parameter(self.pbj_)
        self.wjy_ = F.parameter(self.pwjy_)
        self.by_ = F.parameter(self.pby_)
        self.feed_ = F.zeros([self.embed_size_])
        self.trg_lstm_.init(
            self.src_fw_lstm_.get_c() + self.src_bw_lstm_.get_c(),
            self.src_fw_lstm_.get_h() + self.src_bw_lstm_.get_h())

    # One step decoding.
    def decode_step(self, trg_words, train):
        e = F.pick(self.trg_lookup_, trg_words, 1)
        e = F.dropout(e, self.dropout_rate_, train)
        h = self.trg_lstm_.forward(F.concat([e, self.feed_], 0))
        h = F.dropout(h, self.dropout_rate_, train)
        atten_probs = F.softmax(self.t_concat_fb_ @ h, 0)
        c = self.concat_fb_ @ atten_probs
        self.feed_ = F.tanh(self.whj_ @ F.concat([h, c], 0) + self.bj_)
        return self.wjy_ @ self.feed_ + self.by_

    # Calculates the loss function over given target sentences.
    def loss(self, trg_batch, train):
        losses = []
        for i in range(len(trg_batch)-1):
            y = self.decode_step(trg_batch[i], train)
            loss = F.softmax_cross_entropy(y, trg_batch[i+1], 0)
            losses.append(loss)
        return F.batch.mean(F.sum(losses))


# Training encode decode model.
def train(encdec, trainer, prefix, best_valid_ppl):
    # Registers all parameters to the trainer.
    encdec.register_training(trainer)

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
        print("  learning rate scale = %.4e" % trainer.get_learning_rate_scaling())

        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0.
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = train_ids[ofs:min(ofs+BATCH_SIZE, num_train_sents)]
            src_batch = make_batch(train_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(train_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, True)
            loss = encdec.loss(trg_batch, True)
            train_loss += loss.to_float() * len(batch_ids)

            trainer.reset_gradients()
            loss.backward()
            trainer.update()

        train_ppl = math.exp(train_loss / num_train_labels)
        print("  train PPL = %.4f" % train_ppl)

        # Validation.
        valid_loss = 0.
        for ofs in range(0, num_valid_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = valid_ids[ofs:min(ofs+BATCH_SIZE, num_valid_sents)]
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

        # Saves best model/trainer.
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  saving model/trainer ... ", end="")
            sys.stdout.flush()
            encdec.save(prefix+".")
            trainer.save(prefix+".trainer.config")
            save_ppl(prefix+".valid_ppl.config", best_valid_ppl)
            print("done.")
        else:
            # Learning rate decay by 1/sqrt(2)
            new_scale = .7071 * trainer.get_learning_rate_scaling()
            trainer.set_learning_rate_scaling(new_scale)


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
        if len(trg_ids) > GENERATION_LIMIT+1:
            print("Warning: Sentence generation did not finish in",
                    GENERATION_LIMIT, "iterations.", file=sys.stderr)
            trg_ids.append(eos_ids)
            break
        y = encdec.decode_step(trg_ids[-1], False)
        trg_ids.append(np.array(y.argmax(0)).T)

    return [hyp[:np.where(hyp == eos_id)[0][0]] for hyp in np.array(trg_ids[1:]).T]


# Generates translation by consuming stdin.
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
    parser.add_argument("mode")
    parser.add_argument("model_prefix")
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

    dev = D.Naive() # = D.CUDA(0)
    Device.set_default(dev)
    print("done.", file=sys.stderr)

    if mode == "train":
        encdec = EncoderDecoder("encdec", SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, DROPOUT_RATE)
        trainer = T.Adam()
        trainer.set_weight_decay(1e-6)
        trainer.set_gradient_clipping(5)
        train(encdec, trainer, prefix, 1e10)
    elif mode == "resume":
        print("loading model/trainer ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = EncoderDecoder.load("encdec", prefix+".")
        trainer = Trainer.load(prefix + ".trainer.config")
        valid_ppl = load_ppl(prefix + ".valid_ppl.config")
        print("done.", file=sys.stderr)
        train(encdec, trainer, prefix, valid_ppl)
    else:
        print("loading model ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = EncoderDecoder.load("encdec", prefix+".")
        print("done.", file=sys.stderr)
        test(encdec)

if __name__ == "__main__":
    main()
