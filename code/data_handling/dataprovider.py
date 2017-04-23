'''
Provide processed data ready for consumption by tensorflow model
Return data always as numpy array
'''

import numpy as np
import os
from collections import Counter

# TODO: batching

def clean_line(line_words):
    line_clean = np.empty(30, dtype="<U50")
    line_clean[0] = '<bos>'
    line_clean[29] = '<eos>'
    num_words = len(line_words)
    for i in range(num_words):
        line_clean[i+1] = line_words[i]
    for i in range(28-num_words):
        line_clean[i+num_words+1] = '<pad>'
    return line_clean

def build_vocabulary(data_train):
    word_list = data_train.ravel()
    counts = Counter(word_list)
    relevant_counts = counts.most_common(20000)
    word_ids_list = [(relevant_counts[i][0], i) for i in range(len(relevant_counts))]
    word_dict = dict(word_ids_list)
    return word_dict

def build_targets(data_train, vocab):
    # TODO: somehow build one-hot vectors
    targets = np.empty((len(data_train), 30, len(vocab)+1), dtype=float) # how the fuck should this work??
    for i in range(len(data_train)):
        sentence = data_train[i]
        sentence_matrix = np.empty((30, len(vocab)+1), dtype=float)
        for word in sentence:
            word_1hot = np.empty(len(vocab)+1, dtype=float)
            if word in vocab.keys():
                word_1hot[vocab[word]] = 1.0
            else:
                word_1hot[len(vocab)] = 1.0
            sentence_matrix.append(word_1hot)
        targets[i] = sentence_matrix
    return targets

def process_file(file):
    data_train_list = []
    for line in file.readlines():
        line_words = line.split()
        if len(line_words) < 29:
            data_train_list.append(clean_line(line_words))
    file.close()
    return np.array(data_train_list)

def get_data(DATA_DIR, FILE_BASE, dataset='train'):
    with open(os.path.join(DATA_DIR, FILE_BASE + dataset)) as file:
        data_train = process_file(file)
        vocab = build_vocabulary(data_train)
        targets = build_targets(data_train, vocab)
        return data_train, vocab, targets