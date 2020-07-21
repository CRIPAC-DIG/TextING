import os
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm


if len(sys.argv) < 2:
	sys.exit("Use: python build_graph.py <dataset>")

# settings
datasets = ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']

dataset = sys.argv[1]
if dataset not in datasets:
    sys.exit("wrong dataset name")

try:
    window_size = int(sys.argv[2])
except:
    window_size = 3
    print('using default window size = 3')

try:
    weighted_graph = bool(sys.argv[3])
except:
    weighted_graph = False
    print('using default unweighted graph')

truncate = False # whether to truncate long document
MAX_TRUNC_LEN = 350


print('loading raw data')

# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float,data[1:]))


# load document list
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('data/' + dataset + '.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")

        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())


# load raw text
doc_content_list = []

with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())


# map and shuffle
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids


shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])


# build corpus vocabulary
word_set = set()

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i


# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)


# build label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)


# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)


# build graph function
def build_graph(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()

    for i in tqdm(range(start, end)):
        doc_words = shuffle_doc_words_list[i].split()
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
    
        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])

        x_adj.append(adj)
        x_feature.append(features)

    # one-hot labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    
    return x_adj, x_feature, y, doc_len_list, vocab_set


print('building graphs for training')
x_adj, x_feature, y, _, _ = build_graph(start=0, end=real_train_size)
print('building graphs for training + validation')
allx_adj, allx_feature, ally, doc_len_list_train, vocab_train = build_graph(start=0, end=train_size)
print('building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test = build_graph(start=train_size, end=train_size+test_size)
doc_len_list = doc_len_list_train + doc_len_list_test


# statistics
print('max_doc_length',max(doc_len_list),'min_doc_length',min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
print('training_vocab',len(vocab_train),'test_vocab',len(vocab_test),
      'intersection',len(vocab_train & vocab_test))


# dump objects
with open("data/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("data/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open("data/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.allx_adj".format(dataset), 'wb') as f:
    pkl.dump(allx_adj, f)

with open("data/ind.{}.allx_embed".format(dataset), 'wb') as f:
    pkl.dump(allx_feature, f)

with open("data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)
