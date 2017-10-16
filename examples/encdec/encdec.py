#!/usr/bin/env python3

import sys
import random
import math

from primitiv import Device, Parameter, Graph, Trainer

from primitiv import devices as D
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import trainers as T

from lstm import LSTM
from utils import (
    make_vocab, load_corpus, count_labels, make_batch,
    save_ppl, make_inv_vocab, line_to_sent, argmax, load_ppl
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
DROPOUT_RATE = 0.5
GENERATION_LIMIT = 32

SRC_TRAIN_FILE = "data/train.en"
TRG_TRAIN_FILE = "data/train.ja"
SRC_VALID_FILE = "data/dev.en"
TRG_VALID_FILE = "data/dev.ja"
SRC_TEST_FILE = "data/test.en"
TRG_TEST_FILE = "data/test.ja"

# Encoder-decoder translation model.

class EncoderDecoder(object):
    def __init__(self, name, src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_rate):
        self.name_ = name
        self.dropout_rate_ = dropout_rate
        self.psrc_lookup_ = Parameter(name + "_src_lookup", [embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup_ = Parameter(name + "_trg_lookup", [embed_size, trg_vocab_size], I.XavierUniform())
        self.pwhy_ = Parameter(name + "_why", [trg_vocab_size, hidden_size], I.XavierUniform())
        self.pby_ = Parameter(name + "_by", [trg_vocab_size], I.Constant(0))
        self.src_lstm_ = LSTM(name + "_src_lstm", embed_size, hidden_size)
        self.trg_lstm_ = LSTM(name + "_trg_lstm", embed_size, hidden_size)

    # Loads all parameters.
    @staticmethod
    def load(name, prefix):
        encdec = EncoderDecoder.__new__(EncoderDecoder)
        encdec.name_ = name
        encdec.psrc_lookup_ = Parameter.load(prefix + name + "_src_lookup.param")
        encdec.ptrg_lookup_ = Parameter.load(prefix + name + "_trg_lookup.param")
        encdec.pwhy_ = Parameter.load(prefix + name + "_why.param")
        encdec.pby_ = Parameter.load(prefix + name + "_by.param")
        encdec.src_lstm_ = LSTM.load(name + "_src_lstm", prefix)
        encdec.trg_lstm_ = LSTM.load(name + "_trg_lstm", prefix)
        with open(prefix + name + ".config", "r") as ifs:
            encdec.dropout_rate_ = float(ifs.readline())
        return encdec

    # Saves all parameters.
    def save(self, prefix):
        self.psrc_lookup_.save(prefix + self.name_ + "_src_lookup.param")
        self.ptrg_lookup_.save(prefix + self.name_ + "_trg_lookup.param")
        self.pwhy_.save(prefix + self.name_ + "_why.param")
        self.pby_.save(prefix + self.name_ + "_by.param")
        self.src_lstm_.save(prefix)
        self.trg_lstm_.save(prefix)
        with open(prefix + self.name_ + ".config", "w") as ofs:
            print(self.dropout_rate_, file=ofs)

    # Adds parameters to the trainer.
    def register_training(self, trainer):
        trainer.add_parameter(self.psrc_lookup_)
        trainer.add_parameter(self.ptrg_lookup_)
        trainer.add_parameter(self.pwhy_)
        trainer.add_parameter(self.pby_)
        self.src_lstm_.register_training(trainer)
        self.trg_lstm_.register_training(trainer)

    # Encodes source sentences and prepare internal states.
    def encode(self, src_batch, train):
        # Reversed encoding.
        src_lookup = F.parameter(self.psrc_lookup_)
        self.src_lstm_.init()
        for it in src_batch:
            x = F.pick(src_lookup, it, 1)
            x = F.dropout(x, self.dropout_rate_, train)
            self.src_lstm_.forward(x)

        # Initializes decoder states.
        self.trg_lookup_ = F.parameter(self.ptrg_lookup_)
        self.why_ = F.parameter(self.pwhy_)
        self.by_ = F.parameter(self.pby_)
        self.trg_lstm_.init(self.src_lstm_.get_c(), self.src_lstm_.get_h())

    # One step decoding.
    def decode_step(self, trg_words, train):
        x = F.pick(self.trg_lookup_, trg_words, 1)
        x = F.dropout(x, self.dropout_rate_, train)
        h = self.trg_lstm_.forward(x)
        h = F.dropout(h, self.dropout_rate_, train)
        return self.why_ @ h + self.by_

    # Calculates the loss function over given target sentences.
    def loss(self, trg_batch, train):
        losses = []
        for i in range(len(trg_batch) - 1):
            y = self.decode_step(trg_batch[i], train)
            losses.append(F.softmax_cross_entropy(y, trg_batch[i + 1], 0))
        return F.batch.mean(F.sum(losses))


# Training encoder decoder model.
def train(encdec, trainer, prefix, best_valid_ppl):
    # Registers all parameters to the trainer.
    encdec.register_training(trainer)

    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)
    print("#src_vocab:", len(src_vocab))  # == SRC_VOCAB_SIZE
    print("#trg_vocab:", len(trg_vocab))  # == TRG_VOCAB_SIZE

    # Loads all corpus.
    train_src_corpus = load_corpus(SRC_TRAIN_FILE, src_vocab)
    train_trg_corpus = load_corpus(TRG_TRAIN_FILE, trg_vocab)
    valid_src_corpus = load_corpus(SRC_VALID_FILE, src_vocab)
    valid_trg_corpus = load_corpus(TRG_VALID_FILE, trg_vocab)
    num_train_sents = len(train_trg_corpus)
    num_valid_sents = len(valid_trg_corpus)
    num_train_labels = count_labels(train_trg_corpus)
    num_valid_labels = count_labels(valid_trg_corpus)
    print("train:", num_train_sents, "sentences,", num_train_labels, "labels")
    print("valid:", num_valid_sents, "sentences,", num_valid_labels, "labels")

    # Sentence IDs.
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    # Train/valid loop.
    for epoch in range(MAX_EPOCH):
        # Computation graph.
        g = Graph()
        Graph.set_default(g)

        print("epoch %d/%d:" % (epoch + 1, MAX_EPOCH))
        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0;
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = train_ids[ofs : min(ofs + BATCH_SIZE, num_train_sents)]
            src_batch = make_batch(train_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(train_trg_corpus, batch_ids, trg_vocab)
            trainer.reset_gradients()

            g.clear()

            encdec.encode(src_batch, True)
            loss = encdec.loss(trg_batch, True)
            train_loss += loss.to_list()[0] * len(batch_ids)
            g.backward(loss)
            trainer.update()

        train_ppl = math.exp(train_loss / num_train_labels)
        print("  train PPL = %.4f" % train_ppl)

        # Validation.
        valid_loss = 0
        for ofs in range(0, num_valid_sents, BATCH_SIZE):
            print("%d" % ofs, end="\r")
            sys.stdout.flush()

            batch_ids = valid_ids[ofs : min(ofs + BATCH_SIZE, num_valid_sents)]
            src_batch = make_batch(valid_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(valid_trg_corpus, batch_ids, trg_vocab)

            g.clear()

            encdec.encode(src_batch, False)
            loss = encdec.loss(trg_batch, False)
            valid_loss += loss.to_list()[0] * len(batch_ids)

        valid_ppl = math.exp(valid_loss / num_valid_labels)
        print("  valid PPL = %.4f" % valid_ppl)

        # Calculates test BLEU.
        stats = defaultdict(int)
        with open(SRC_TEST_FILE, "r") as fp_src, open(TRG_TEST_FILE, "r") as fp_trg:
            for ofs, (test_src, test_ref) in enumerate(zip(fp_src, fp_trg)):
                print("%d" % ofs, end="\r")
                sys.stdout.flush()

                hyp_ids = test_one(encdec, src_vocab, trg_vocab, test_src)
                hyp = [inv_trg_vocab[wid] for wid in hyp_ids]
                for k, v in get_bleu_stats(test_ref.split(), hyp).items():
                    stats[k] += v

        bleu = calculate_bleu(stats)
        print("  test BLEU = %.2f" % (100 * bleu))

        # Saves best model/trainer.
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("  saving model/trainer ... ", end="")
            sys.stdout.flush()
            encdec.save(prefix + '.')
            trainer.save(prefix + ".trainer.config")
            save_ppl(prefix + ".valid_ppl.config", best_valid_ppl)
            print("done.")


def test_one(encdec, src_vocab, trg_vocab, line):
    g = Graph()
    Graph.set_default(g)

    src_corpus = [line_to_sent(line.strip(), src_vocab)]
    src_batch = make_batch(src_corpus, [0], src_vocab)

    encdec.encode(src_batch, False)

    # Generates target words one-by-one.
    trg_ids = [trg_vocab["<bos>"]]
    eos_id = trg_vocab["<eos>"]
    while trg_ids[-1] != eos_id:
        if len(trg_ids) > GENERATION_LIMIT + 1:
            print("Warning: Sentence generation did not finish in", GENERATION_LIMIT, "iterations.", file=sys.stderr)
            trg_ids.append(eos_id)
            break
        y = encdec.decode_step([trg_ids[-1]], False)
        logits = y.to_list()
        trg_ids.append(argmax(logits))

    return trg_ids[1:-1]


# Generates translation by consuming stdin.
def test(encdec):
    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)

    for line in sys.stdin:
        trg_ids = test_one(encdec, src_vocab, trg_vocab, inv_trg_vocab, line)
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
        encdec = EncoderDecoder.load("encdec", prefix + '.')
        trainer = Trainer.load(prefix + ".trainer.config")
        valid_ppl = load_ppl(prefix + ".valid_ppl.config")
        print("done.", file=sys.stderr)
        train(encdec, trainer, prefix, valid_ppl)
    else:  # mode == "test"
        print("loading model ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = EncoderDecoder.load("encdec", prefix + '.');
        print("done.", file=sys.stderr)
        test(encdec);


if __name__ == "__main__":
    main()
