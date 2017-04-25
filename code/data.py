'''
Provide processed data ready for consumption by tensorflow model
Return data always as numpy array
'''

import numpy as np
import os
from collections import Counter

class NlpData:

    data = None
    data_only_ids = None
    vocab = None
    current_batch = 0

    def get_next_batch(self, batch_size):
        lower = self.current_batch * batch_size
        upper = (self.current_batch + 1) * batch_size
        self.current_batch += 1
        if lower >= len(self.data):
            return None
        if upper < len(self.data):
            return self.data[lower:upper]
        else:
            upper = len(self.data)-1
            temp_data = self.data[lower:upper]
            diff = batch_size - (upper - lower + 1)
            for i in range(diff):
                temp_data.append(temp_data[upper])
            return temp_data

    def clean_line(self, line_words):
        line_clean = np.empty(30, dtype="<U50")
        line_clean[0] = '<bos>'
        line_clean[29] = '<eos>'
        num_words = len(line_words)
        for i in range(num_words):
            line_clean[i+1] = line_words[i]
        for i in range(28-num_words):
            line_clean[i+num_words+1] = '<pad>'
        return line_clean

    def build_vocabulary(self, vocab_size):
        word_list = self.data.ravel()
        counts = Counter(word_list)
        relevant_counts = counts.most_common(vocab_size-1) # -1 because of <unk>
        word_ids_list = [(relevant_counts[i][0], i) for i in range(len(relevant_counts))]
        word_dict = dict(word_ids_list)
        self.vocab = word_dict

    def process_file(self, file):
        data_list = []
        for line in file.readlines():
            line_words = line.split()
            if len(line_words) < 30:
                data_list.append(self.clean_line(line_words))
        file.close()
        self.data = np.array(data_list)
    
    def generate_data_only_ids():
        if self.data is None or self.vocab is None:
            raise InvalidOperationException("The data needs to initialized first")
        temp_data = []
        for sentence in self.data:
            sentence = []
            for word in sentence:
                sentence.append(self.vocab[word])
            temp_data.append(np.array(sentence))
        self.data_only_ids = np.array(temp_data)

    def generate_submission():
        # TODO: generate the submission file
        return

    def __init__(self, data_dir, file_name, vocab_size):
        with open(os.path.join(data_dir, file_name)) as file:
            self.process_file(file)
            self.build_vocabulary(vocab_size)
            self.generate_data_only_ids()