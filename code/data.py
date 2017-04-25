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
        """Returns the next batch of samples"""
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

    def __clean_line(self, line_words, seq_length):
        line_clean = ["" for i in range(seq_length)]
        line_clean[0] = '<bos>'
        line_clean[seq_length-1] = '<eos>'
        num_words = len(line_words)
        for i in range(num_words):
            line_clean[i+1] = line_words[i]
        for i in range((seq_length - 2) - num_words):
            line_clean[i+num_words+1] = '<pad>'
        return line_clean

    def __build_vocabulary(self, vocab_size):
        word_list = [word for sentence in self.data for word in sentence]
        counts = Counter(word_list)
        relevant_counts = counts.most_common(vocab_size-1) # -1 because of <unk>
        word_ids_list = [(relevant_counts[i][0], i) for i in range(len(relevant_counts))]
        word_dict = dict(word_ids_list)
        self.vocab = word_dict

    def __process_file(self, file, seq_length):
        data_list = []
        for line in file.readlines()[:256]:
            line_words = line.split()
            if len(line_words) < 30:
                data_list.append(self.__clean_line(line_words, seq_length))
        file.close()
        self.data = data_list
    
    def __generate_data_only_ids(self):
        if self.data is None or self.vocab is None:
            raise InvalidOperationException("The data needs to initialized first")
        temp_data = []
        for sentence in self.data:
            sentence = []
            for word in sentence:
                sentence.append(self.vocab[word])
            temp_data.append(sentence)
        self.data_only_ids = temp_data

    def generate_submission(self):
        """Generates submission for testing system"""
        # TODO: generate the submission file
        return

    def __init__(self, data_dir, file_name, vocab_size, seq_length):
        with open(os.path.join(data_dir, file_name)) as file:
            self.__process_file(file, seq_length)
            self.__build_vocabulary(vocab_size)
            self.__generate_data_only_ids()