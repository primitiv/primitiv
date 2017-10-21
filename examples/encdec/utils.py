from collections import defaultdict


# Gathers the set of words from space-separated corpus and makes a vocabulary.
def make_vocab(path, size):
    if (size < 3):
        print("Vocab size should be >= 3.", file=sys.stderr)
        sys.exit(1)
    ifs = open(path, "r")

    # Counts all word existences.
    freq = defaultdict(lambda : 0)
    for line in ifs:
        for word in line.split():
            freq[word] += 1

    # Sorting.
    # Chooses top size-3 frequent words to make the vocabulary.
    vocab = {}
    vocab["<unk>"] = 0
    vocab["<bos>"] = 1
    vocab["<eos>"] = 2
    for i, (k, v) in zip(range(3, size), sorted(freq.items(), key=lambda x: -x[1])):
        vocab[k] = i
    return vocab

# Generates ID-to-word dictionary.
def make_inv_vocab(vocab):
    ret = [k for k, v in sorted(vocab.items(), key=lambda x: x[1])]
    return ret

# Generates word ID list from a sentence.
def line_to_sent(line, vocab):
    unk_id = vocab["<unk>"]
    converted = "<bos> " + line + " <eos>"
    return [vocab.get(word, unk_id) for word in converted.split()]

# Generates word ID list from a corpus.
# All out-of-vocab words are replaced to <unk>.
def load_corpus(path, vocab):
    return [line_to_sent(line, vocab) for line in open(path)]

# Generates word ID list from a reference sentence.
def line_to_sent_ref(line, vocab):
    # NOTE(odashi):
    # -1 never becomes a word ID of any specific words and this is useful to
    # prevent BLEU contamination.
    unk_id = -1
    converted = "<bos> " + line + " <eos>"
    return [vocab.get(word, unk_id) for word in converted.split()]

# Generates word ID list from a reference corpus.
# All out-of-vocab words are replaced to -1.
def load_corpus_ref(path, vocab):
    return [line_to_sent_ref(line, vocab) for line in open(path)]

# Counts output labels in the corpus.
def count_labels(corpus):
    ret = 0
    for sent in corpus:
        ret += len(sent) - 1  # w/o <bos>
    return ret

# Extracts a minibatch from loaded corpus
# NOTE(odashi):
# Lengths of all sentences are adjusted to the maximum one in the minibatch.
# All additional subsequences are filled by <eos>. E.g.,
#   input: {
#     {<bos>, w1, <eos>},
#     {<bos>, w1, w2, w3, w4, <eos>},
#     {<bos>, w1, w2, <eos>},
#     {<bos>, w1, w2, w3, <eos>},
#   }
#   output: {
#     {<bos>, <bos>, <bos>, <bos>},
#     {   w1,    w1,    w1,    w1},
#     {<eos>,    w2,    w2,    w2},
#     {<eos>,    w3, <eos>,    w3},
#     {<eos>,    w4, <eos>, <eos>},
#     {<eos>, <eos>, <eos>, <eos>},
#   }
def make_batch(corpus, sent_ids, vocab):
    batch_size = len(sent_ids)
    eos_id = vocab["<eos>"]
    max_len = 0
    for sid in sent_ids:
        max_len = max(max_len, len(corpus[sid]))
    batch = [[eos_id] * batch_size for i in range(max_len)]
    for i in range(batch_size):
        sent = corpus[sent_ids[i]]
        for j in range(len(sent)):
            batch[j][i] = sent[j]
    return batch


# Helper to save current ppl.
def save_ppl(path, ppl):
    with open(path, "w") as ofs:
        print(ppl, file=ofs)


# Helper to load last ppl.
def load_ppl(path):
    with open(path, "r") as ifs:
        return float(ifs.readline());
