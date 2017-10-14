#!/usr/bin/env python3
# coding: utf-8

import sys
import random
import math

from primitiv import Device, Parameter, Graph
from primitiv.devices import Naive

from primitiv import operators as F
from primitiv import initializers as I
from primitiv import trainers as T

from lstm import LSTM
from utils import (
    make_vocab, load_corpus, count_labels, make_batch,
    save_ppl, make_inv_vocab, line_to_sent, argmax
)

from argparse import ArgumentParser

SRC_VOCAB_SIZE = 4000;
TRG_VOCAB_SIZE = 5000;
NUM_EMBED_UNITS = 512;
NUM_HIDDEN_UNITS = 512;
BATCH_SIZE = 64;
MAX_EPOCH = 100;
GENERATION_LIMIT = 32;
DROPOUT_RATE = 0.5;

SRC_TRAIN_FILE = "data/train.en";
TRG_TRAIN_FILE = "data/train.ja";
SRC_VALID_FILE = "data/dev.en";
TRG_VALID_FILE = "data/dev.ja";

# Encoder-decoder translation model with dot-attention.
class EncoderDecoder(object):
    def __init__(self, name, src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_rate):
        self.name_ = name
        self.embed_size_ = embed_size
        self.dropout_rate_ = dropout_rate
        self.psrc_lookup_ = Parameter(name+"_src_lookup", [embed_size, src_vocab_size], I.XavierUniform())
        self.ptrg_lookup_ = Parameter(name+"_trg_lookup", [embed_size, trg_vocab_size], I.XavierUniform())
        self.pwhj_ = Parameter(name+"_whj", [embed_size, 2*hidden_size], I.XavierUniform())
        self.pbj_ = Parameter(name+"_pbj", [embed_size], I.Constant(0))
        self.pwjy_ = Parameter(name+"_wjy", [trg_vocab_size, embed_size], I.XavierUniform())
        self.pby_ = Parameter(name+"_pby", [trg_vocab_size], I.Constant(0))
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
        e_list = list()
        for x in src_batch:
            e = F.pick(src_lookup, x, 1)
            e = F.dropout(e, self.dropout_rate_, train)
            e_list.append(e)

        # Forward encoding
        self.src_fw_lstm_.init()
        f_list = list()
        for e in e_list:
            f = self.src_fw_lstm_.forward(e)
            f = F.dropout(f, self.dropout_rate_, train)
            f_list.append(f)

        # Backward encoding
        self.src_bw_lstm_.init()
        b_list = list()
        for e in reversed(e_list):
            b = self.src_bw_lstm_.forward(e)
            b = F.dropout(b, self.dropout_rate_, train)
            b_list.append(b)
        b_list.reverse()

        # Concatenates RNN states.
        fb_list = list()
        for i in range(len(src_batch)):
            fb_list.append(f_list[i]+b_list[i])
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
    def decode_step(self, trg_word, train):
        e = F.pick(self.trg_lookup_, trg_word, 1)
        e = F.dropout(e, self.dropout_rate_, train)
        h = self.trg_lstm_.forward(F.concat([e, self.feed_], 0))
        h = F.dropout(h, self.dropout_rate_, train)
        atten_probs = F.softmax(self.t_concat_fb_ @ h, 0)
        c = self.concat_fb_ @ atten_probs
        self.feed_ = F.tanh(self.whj_ @ F.concat([h, c], 0) + self.bj_)
        return self.wjy_ @ self.feed_ + self.by_

    # Caluculates the loss function over given target sentences.
    def loss(self, trg_batch, train):
        losses = list()
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
    print("#src_vocab:", len(src_vocab))
    print("#trg_vocab:", len(trg_vocab))

    # Loads all corpus
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

    # Sentence IDs
    train_ids = list(range(num_train_sents))
    valid_ids = list(range(num_valid_sents))

    # Computation graph.
    g = Graph()
    Graph.set_default(g)

    # Train/valid loop.
    for epoch in range(MAX_EPOCH):
        print("epoch {}/{}, lr_scale = {}".format(
            epoch+1, MAX_EPOCH, trainer.get_learning_rate_scaling()))
        # Shuffles train sentence IDs.
        random.shuffle(train_ids)

        # Training.
        train_loss = 0.
        for ofs in range(0, num_train_sents, BATCH_SIZE):
            batch_ids = train_ids[ofs:min(ofs+BATCH_SIZE, num_train_sents)]
            src_batch = make_batch(train_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(train_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, True)
            loss = encdec.loss(trg_batch, True)
            train_loss += g.forward(loss).to_list()[0] * len(batch_ids)

            trainer.reset_gradients()
            g.backward(loss)
            trainer.update()

            print("\r%d" % (ofs), end="")
            sys.stdout.flush()

        train_ppl = math.exp(train_loss / num_valid_labels)
        print("\n\ttrain ppl =", train_ppl)

        # Valdation.
        valid_loss = 0.
        for ofs in range(0, num_valid_sents, BATCH_SIZE):
            batch_ids = valid_ids[ofs:min(ofs+BATCH_SIZE, num_valid_sents)]
            src_batch = make_batch(valid_src_corpus, batch_ids, src_vocab)
            trg_batch = make_batch(valid_trg_corpus, batch_ids, trg_vocab)

            g.clear()
            encdec.encode(src_batch, False)
            loss = encdec.loss(trg_batch, False)
            valid_loss += g.forward(loss).to_list()[0] * len(batch_ids)

            print("\r%d"%(ofs), end="")
            sys.stdout.flush()

        valid_ppl = math.exp(valid_loss/num_valid_labels)
        print("\n\tvalid ppl =", valid_ppl)

        # Saves best model/trainer.
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            print("\tsaving model/trainer ...", end="")
            sys.stdout.flush()
            encdec.save(prefix+".")
            trainer.save(prefix+".trainer.config")
            save_ppl(prefix+".valid_ppl.config", best_valid_ppl)
            print("done.")
        else:
            # Learning rate decay by 1/sqrt(2)
            new_scale = .7071 * trainer.get_learning_rate_scaling()
            trainer.set_learning_rate_scaling(new_scale)

# Generates translation by consuming stdin.
def test(encdec):
    g = Graph()
    Graph.set_default(g)

    # Loads vocab.
    src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE)
    trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE)
    inv_trg_vocab = make_inv_vocab(trg_vocab)

    for line in sys.stdin:
        src_corpus = [line_to_sent(line.strip(), src_vocab)]
        src_batch = make_batch(src_corpus, [0], src_vocab)
        encdec.encode(src_batch, False)

        # Generates target words one-by-one.
        trg_ids = [trg_vocab["<bos>"]]
        eos_id = trg_vocab["<eos>"]
        while trg_ids[-1] != eos_id:
            if len(trg_ids) > GENERATION_LIMIT+1:
                print("Waring: Sentene generation did not finish in",
                      GENERATION_LIMIT, "iterations.", file=sys.stderr)
                trg_ids.append(eos_id)
                break
            y = encdec.decode_step([trg_ids[-1]], False)
            logits = y.to_list()
            trg_ids.append(argmax(logits))

        # Prints the result.
        print(" ".join(inv_trg_vocab[wid] for wid in trg_ids[1:-1]))

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

    dev = Naive() # = CUDA(0)
    Device.set_default(dev)
    print("done.", file=sys.stderr)

    if mode == "train":
        encdec = EncoderDecoder("encdec", SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, DROPOUT_RATE)
        trainer = T.Adam()
        trainer.set_weight_decay(1e-6)
        trainer.set_gradient_clipping(5)
        train(encdec, trainer, prefix, 1e10)
    elif mode == "resume":
        print("loading mode/trainer ... ", end="", file=sys.stderr)
        sys.stderr.flush()
        encdec = EncoderDecoder.load("encdec", prefix+".")
        trainer = Trainer.load(prefix + ".trainer.config")
        valid_ppl = load_ppl(prefix + ".valid_ppl.config")
        print("done.", file=sys.stderr)
        train(encdec, trainer, prefix, valid_ppl)
    else:
        print("loading mode ... ", file=sys.stderr)
        encdec = EncoderDecoder.load("encdec", prefix+".")
        print("done.", file=sys.stderr)
        test(encdec)

if __name__ == '__main__':
    main()
