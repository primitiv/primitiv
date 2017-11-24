# Common utility functions for PTB examples.

# Gathers the set of words from space-separated corpus.
def make_vocab(filename):
    vocab = {}
    with open(filename, "r") as ifs:
        for line in ifs:
            line = "<s> " + line.strip() + " <s>"
            for word in line.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab


# Generates word ID list using corpus and vocab.
def load_corpus(filename, vocab):
    corpus = []
    with open(filename, "r") as ifs:
        for line in ifs:
            line = "<s> " + line.strip() + " <s>"
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
