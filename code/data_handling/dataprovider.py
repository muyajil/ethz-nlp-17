'''
Provide processed data ready for consumption by tensorflow model
Return data always as numpy array
'''

import numpy as np
import os
from collections import Counter

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
    return [a for (a, b) in relevant_counts]

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
        return process_file(file)