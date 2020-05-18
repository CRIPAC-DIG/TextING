import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
from tqdm import tqdm

if len(sys.argv) < 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['mr', 'ohsumed', 'R8', 'R52', '20ng', 'ag_news']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

print('loading raw data')

word_embeddings_dim = 300
word_vector_map = {}

embeddings = {}
with open('glove.6B.300d.txt', 'r') as f:
    for line in f.readlines():
        data = line.split()
        embeddings[str(data[0])] = list(map(float,data[1:]))

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()
# print(doc_train_list)
# print(doc_test_list)

doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()
# print(doc_content_list)

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
# print('train_ids', train_ids)
random.shuffle(train_ids)

# partial labeled data
# train_ids = train_ids[:int(0.2 * len(train_ids))]

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
# print('test_ids', test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
# print('ids', ids)
# print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

# out-of-vocabulary word embeddings initialize
oov = {}
for v in vocab:
    if v not in oov:
        oov[v] = list(np.random.uniform(-0.01,0.01,300))

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()


# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)


label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# select 90% training set
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

# ---------------------------------------------------------------

try:
    window_size = int(sys.argv[2])
except:
    window_size = 3
    print('using default window size = 3')

x_adj = []
x_feature = []
doc_len_list = []

print('building graphs for training')
for i in tqdm(range(real_train_size)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    doc_vocab = set()

    for word in words:
        doc_vocab.add(word)
    doc_vocab = list(doc_vocab)
    doc_nodes = len(doc_vocab)
    doc_len_list.append(doc_nodes)

    doc_word_id = {}
    for j in range(len(doc_vocab)):
        doc_word_id[doc_vocab[j]] = j

    windows = []
    if doc_len <= window_size:
        windows.append(words)
    else:
        for j in range(doc_len - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
    
    freq = {}
    for window in windows:
        appeared = set()
        for j in range(len(window)):
            if window[j] in appeared:
                continue
            if window[j] in freq:
                freq[window[j]] += 1
            else:
                freq[window[j]] = 1
            appeared.add(window[j])

    count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = word_id_map[word_p]
                word_q = window[q]
                word_q_id = word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_str = str(word_p_id) + ',' + str(word_q_id)
                if word_pair_str in count:
                    count[word_pair_str] += 1
                else:
                    count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_q_id) + ',' + str(word_p_id)
                if word_pair_str in count:
                    count[word_pair_str] += 1
                else:
                    count[word_pair_str] = 1
    
    row = []
    col = []
    weight = []
    features = []
    #num_window = len(windows)
    for key in count:
        temp = key.split(',')
        p = int(temp[0])
        q = int(temp[1])
        row.append(doc_word_id[vocab[p]])
        col.append(doc_word_id[vocab[q]])
        weight.append(1.)
    
    for k, v in sorted(doc_word_id.items(), key=lambda x: x[1]):
        features.append(embeddings[k] if k in embeddings else oov[k])

    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    x_adj.append(adj)
    x_feature.append(features)

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
#print('y', y)


# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

allx_adj = []
allx_feature = []
vocab_train = set()

print('building graphs for training + validation')
for i in tqdm(range(train_size)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    doc_vocab = set()

    for word in words:
        doc_vocab.add(word)
        vocab_train.add(word)
    doc_vocab = list(doc_vocab)
    doc_nodes = len(doc_vocab)

    doc_word_id = {}
    for j in range(len(doc_vocab)):
        doc_word_id[doc_vocab[j]] = j

    windows = []
    if doc_len <= window_size:
        windows.append(words)
    else:
        for j in range(doc_len - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

    freq = {}
    for window in windows:
        appeared = set()
        for j in range(len(window)):
            if window[j] in appeared:
                continue
            if window[j] in freq:
                freq[window[j]] += 1
            else:
                freq[window[j]] = 1
            appeared.add(window[j])

    count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = word_id_map[word_p]
                word_q = window[q]
                word_q_id = word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_str = str(word_p_id) + ',' + str(word_q_id)
                if word_pair_str in count:
                    count[word_pair_str] += 1
                else:
                    count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_q_id) + ',' + str(word_p_id)
                if word_pair_str in count:
                    count[word_pair_str] += 1
                else:
                    count[word_pair_str] = 1

    row = []
    col = []
    weight = []
    features = []
    for key in count:
        temp = key.split(',')
        p = int(temp[0])
        q = int(temp[1])
        row.append(doc_word_id[vocab[p]])
        col.append(doc_word_id[vocab[q]])
        weight.append(1.)       
    
    for k, v in sorted(doc_word_id.items(), key=lambda x: x[1]):
        features.append(embeddings[k] if k in embeddings else oov[k])

    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    allx_adj.append(adj)
    allx_feature.append(features)

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

ally = np.array(ally)


# tx: feature vectors of test docs
test_size = len(test_ids)

tx_adj = []
tx_feature = []
doc_len_list = []
vocab_test = set()

print('building graphs for test')
for i in tqdm(range(test_size)):
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    doc_vocab = set()

    for word in words:
        doc_vocab.add(word)
        vocab_test.add(word)
    doc_vocab = list(doc_vocab)
    doc_nodes = len(doc_vocab)
    doc_len_list.append(doc_nodes)

    doc_word_id = {}
    for j in range(len(doc_vocab)):
        doc_word_id[doc_vocab[j]] = j

    windows = []
    if doc_len <= window_size:
        windows.append(words)
    else:
        for j in range(doc_len - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
    
    freq = {}
    for window in windows:
        appeared = set()
        for j in range(len(window)):
            if window[j] in appeared:
                continue
            if window[j] in freq:
                freq[window[j]] += 1
            else:
                freq[window[j]] = 1
            appeared.add(window[j])

    count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = word_id_map[word_p]
                word_q = window[q]
                word_q_id = word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_str = str(word_p_id) + ',' + str(word_q_id)
                if word_pair_str in count:
                    count[word_pair_str] += 1
                else:
                    count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_q_id) + ',' + str(word_p_id)
                if word_pair_str in count:
                    count[word_pair_str] += 1
                else:
                    count[word_pair_str] = 1

    row = []
    col = []
    weight = []
    features = []
    for key in count:
        temp = key.split(',')
        p = int(temp[0])
        q = int(temp[1])
        row.append(doc_word_id[vocab[p]])
        col.append(doc_word_id[vocab[q]])
        weight.append(1.)       

    for k, v in sorted(doc_word_id.items(), key=lambda x: x[1]):
        features.append(embeddings[k] if k in embeddings else oov[k])

    adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    tx_adj.append(adj)
    tx_feature.append(features)

print('max_doc_length',max(doc_len_list),'min_doc_length',min(doc_len_list),
      'average {:.2f}'.format(sum(doc_len_list)/len(doc_len_list)))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
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
