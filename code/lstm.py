import os
import argparse
import numpy as np
from collections import Counter

# Global Constants
DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(DIR, 'data')
FILE_BASE = 'sentences.'
PARSER = argparse.ArgumentParser(description='LSTM Implementation in Tensorflow')

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
    

def main():
    data_train_list = []
    with open(os.path.join(DATA_DIR, FILE_BASE + 'train')) as file:
        for line in file.readlines():
            line_words = line.split()
            if len(line_words) < 29:
                data_train_list.append(clean_line(line_words))
        file.close()
    data_train = np.array(data_train_list)
    vocab = build_vocabulary(data_train)

if __name__ == '__main__':
    main()
