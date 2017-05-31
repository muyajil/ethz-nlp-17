import os
import sys

import numpy as np
from collections import Counter
from itertools import islice
import pandas as pd
import pickle
import time

# TODO: Make the encode/decoder preproc in data_reader easier to modify

def get_curr_time():
    return time.time()

def estimate_time(start_time, end_time=None, multiplier=None):
    if end_time is None: end_time = get_curr_time()
    diff_time = end_time - start_time
    if not multiplier is None: diff_time = diff_time * multiplier
    return format_time(diff_time)

def format_time(time_val):
    m, s = divmod(time_val, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

class Vocab(object):
    def __init__(self):
        '''Stores mappings between words (string) and tokens (integer).
        '''
        self.word_to_index = dict()
        self.index_to_word = dict()
        self.word_counter = Counter()

        self.unknown = '<unk>'
        self.padding = '<pad>'
        self.begin = '<bos>'
        self.end = '<eos>'

        self._is_constructed = False

    def get_vocab_size(self):
        size = len(self.word_to_index)
        assert size == len(self.index_to_word)
        return size

    def update_with_meta(self, mr):
        '''Add meta tokens to the vocabulary.
        '''
        assert self._is_constructed
        print("WARNING: adding %s meta tags to vocabulary, size should be adjusted as necessary!!" %mr.num_genres)
        self.word_counter.update(mr.genre_counter)
        for genre in mr.genre_counter:
            self._insert(genre)
        print("WARNING: vocab size after adding meta tags: %s" %self.get_vocab_size())
        return        

    def update(self, sentence):
        '''Update counts with words in sentence.
        '''
        if isinstance(sentence, str):
            self.word_counter.update(sentence.split())
        else:
            assert isinstance(sentence[0], str)
            self.word_counter.update(sentence)
        return

    def construct(self, path, vocab_size, meta_reader=None):
        '''Construct a vocabulary from path, cutting infrequent words to given size.
        '''
        self._is_constructed = True
        print("Constructing vocabulary using:\n  Path: %s\n  Vocab size: %d" %(os.path.abspath(path), vocab_size))
        if meta_reader is not None:
            print("  Metareader: %s" %os.path.abspath(meta_reader.meta_path))

        for line in open(path, 'r'):
            self.update(line)
        for key in [self.unknown, self.padding, self.begin, self.end]:
            self._insert(key)
        if not meta_reader is None:
            for key in meta_reader.genre_counter:
                self._insert(key)
        if vocab_size is None:
            words = [w for (w, _) in self.word_counter.most_common()]
        else:
            words = [w for (w, _) in self.word_counter.most_common(vocab_size - len(self.word_to_index))]
        for w in words:
            self._insert(w)
        return

    def _insert(self, word):
        '''Stores word in the token maps.
        '''
        if not word in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        return

    def encode(self, word):
        '''Turn token word into integer index.
        '''
        try:
            index = self.word_to_index[word]
        except KeyError:
            index = self.word_to_index[self.unknown]
        return index

    def decode(self, index):
        '''Turn integer index into token word.
        '''
        return self.index_to_word[index]

class MetaReader(object):
    def __init__(self, meta_path, label_path=None):
        '''Holds genre information and maps them to lines in dataset.

        Loads genre information from meta_path
        Loads map to dataset lines from label_path

        self.genres: maps movie id to genres (movies have more than one)
        self.movie_ids: maps line id to movie id
        self.most_common_genre: maps movie id to the single most common genre in self.genres
        '''

        self.meta_path = meta_path
        self.label_path = label_path

        print("Loading genre data using:\n  Path: %s" %os.path.abspath(meta_path))
        raw_genres = pd.read_csv(meta_path, delimiter='\t', usecols=(6,))['genre'].values.astype(str)
        self.genre_counter = Counter()
        self.genres = list()
        for i, line in enumerate(raw_genres):
            genre_list = [self.wrap_genre_text(x.strip()) for x in line.split(',')]
            self.genre_counter.update(genre_list)
            self.genres.append(genre_list)
        self.num_genres = len(self.genre_counter)

        self.most_common_genre = self.collapse_to_most_common_genre()
        self.movie_ids = self.load_map_to_data(label_path) if not label_path is None else None
        return
    
    def wrap_genre_text(self, text):
        return '<%s>' %text.upper()

    def collapse_to_most_common_genre(self):
        most_common_genre = list()
        for genre_list in self.genres:
            most_common = genre_list[np.argmax([self.genre_counter[g] for g in genre_list])]
            most_common_genre.append(most_common)
        return most_common_genre

    def load_map_to_data(self, label_path):
        assert label_path.endswith('Labels.txt')
        print("  Reading labels from: %s" %os.path.abspath(label_path))
        self.label_path = label_path
        # index from zero to make literally everyones life easier
        movie_ids = np.loadtxt(label_path, delimiter='\t', usecols=(0,), dtype=int) - 1

        # 0 for training set, 484 for valid set
        assert movie_ids.min() == 0 or movie_ids.min() == 484
        return movie_ids

    def get_genre(self, line, most_common=False):
        '''Maps line to genre(s).

        line is from data_path corresponding to label_path
        '''
        assert not self.movie_ids is None
        mid = self.movie_ids[line]
        if most_common: return self.most_common_genre[mid]
        return self.genres[mid]
        

class DataReader(object):
    def __init__(self, vocab=None, meta_reader=None):
        self.vocab = vocab
        if not vocab is None: self._store_tokens()
        self.meta_reader = meta_reader
        self.encode_inputs = None
        self.decode_inputs = None
        self.decode_targets = None
        self.line_ids = list()
        self.nexchange = None
        self.sent_size = None
        self.cache_file = '/tmp/nlp-project-sentences-train.pickle'
        return

    def _store_tokens(self):
        self._pad_token = self.vocab.encode(self.vocab.padding)
        self._bos_token = self.vocab.encode(self.vocab.begin)
        self._eos_token = self.vocab.encode(self.vocab.end)
        return

    def construct(self, path, sent_size, vocab_size=None):
        '''Load vocabulary, mapping words to tokens, then load corpus as tokens.
        '''
        #if os.path.isfile(self.cache_file):
        #    print('loading cached data from: %s' % self.cache_file)
        #    with open(self.cache_file, 'rb') as f:
        #        self.data, self.vocab = pickle.load(f)
        #else:


        if not vocab_size is None:
            self.vocab = Vocab()
            self.vocab.construct(path, vocab_size)
            self._store_tokens()
        else:
            assert not self.vocab is None, "Vocabulary must be given"

            # second pass to load corpus as tokens
        self.nexchange = 2 * sum(1 for line in open(path, 'r'))
        self.sent_size = sent_size
            # add padding, bos, eos symbols
        self.encode_inputs = np.full((self.nexchange, sent_size), self._pad_token, dtype=int)
        self.decode_inputs = np.full((self.nexchange, sent_size), self._pad_token, dtype=int)
        self.decode_targets = np.full((self.nexchange, sent_size), self._pad_token, dtype=int)
        print("Reading & tokenizing using:\n  Path: %s\n  Sentence size: %s" %(os.path.abspath(path), sent_size))

        for indx, line in enumerate(open(path, 'r')):
            a, b, c = self._tokenize_line(line)

            encode_first_inputs = self.preproc_encode_input(a, sent_size)
            encode_second_inputs = self.preproc_encode_input(b, sent_size)
            decode_first_inputs, decode_first_targets = self.preproc_decode_input(b, sent_size)
            decode_second_inputs, decode_second_targets  = self.preproc_decode_input(c, sent_size)

            first_indx = 2 * indx
            second_indx = 2 * indx + 1

            # Encode has padding at start, Decode has padding at end
            self.encode_inputs[first_indx, -len(encode_first_inputs): ] = encode_first_inputs
            self.decode_inputs[first_indx, :len(decode_first_inputs)] = decode_first_inputs
            self.decode_targets[first_indx, :len(decode_first_targets)] = decode_first_targets
            
            self.encode_inputs[second_indx, -len(encode_second_inputs):] = encode_second_inputs
            self.decode_inputs[second_indx, :len(decode_second_inputs)] = decode_second_inputs
            self.decode_targets[second_indx, :len(decode_second_targets)] = decode_second_targets

        #with open(self.cache_file, 'wb') as f:
        #    pickle.dump((self.data, self.vocab), f)
        #print('caching data set here: %s' % self.cache_file)
                
        return

    def preproc_encode_input(self, tokens, max_sent_size):
        '''Preprocess the encoder input tokens (e.g. reverse).
        '''
        if not max_sent_size is None:
            tokens = tokens[:max_sent_size]
        encode_tokens = tokens[::-1]
        return encode_tokens

    def preproc_decode_input(self, tokens, max_sent_size):
        '''Preprocess the decoder input tokens (e.g. add bos/eos tokens).
        '''
        decode_inputs = [self._bos_token]
        decode_inputs.extend(tokens)
        decode_inputs = decode_inputs[:max_sent_size]

        decode_targets = list()
        decode_targets.extend(decode_inputs[1:])
        decode_targets.append(self._eos_token)

        #decode_inputs = decode_tokens[:max_sent_size - 1]
        #decode_tokens.append(self._eos_token)
        return decode_inputs, decode_targets

    def _encode_str(self, text):
        '''Encode string into token representation.
        '''
        encode_sent = [self.vocab.encode(word) for word in text.split()]
        return encode_sent

    def _tokenize_line(self, line):
        raw_a, raw_b, raw_c = line.strip().split('\t')
        encode_a = self._encode_str(raw_a)
        encode_b = self._encode_str(raw_b)
        encode_c = self._encode_str(raw_c)
        return encode_a, encode_b, encode_c

    def _shuffle(self):
        '''Shuffle the rows of self.data.
        '''
        assert self.encode_inputs is not None
        assert self.decode_inputs is not None
        assert self.decode_targtes is not None
        perm = np.random.permutation(self.encode_inputs.shape[0])
        assert perm.size == len(self.encode_inputs) == len(self.decode_inputs) == len(self.decode_targets)
        self.encode_inputs = self.encode_inputs[perm]
        self.decode_inputs = self.decode_inputs[perm]
        self.decode_targets = self.decode_targets[perm]
        return

    def get_iterator(self, batch_size, shuffle=True, meta_tokens=None):
        '''Iterator yielding batch number, batch_size sentences.

        meta_tokens can be None, most_common, or all
        if meta_tokens is None, yield:
            batch_id, encoder_inputs_, decoder_inputs_, decoder_targets_
        elif meta_tokens is most_common, yield:
            batch_id, encoder_inputs_, decoder_inputs_, decoder_targets_, encoded_batch_genres_
            batch_genres_ is an arr of size batch_size of genre tags (strings)
        elif meta_tokens is all, yield:
            batch_id, encoder_inputs_, decoder_inputs_, decoder_targets_, encoded_batch_genres_
            encoded_batch_genres_ is an arr of zeros size [batch_size, vocab_size],
                with 1s where the data point has a genre
                (We might just use most_common only ... )
            
        '''
        if shuffle:
            order = np.random.permutation(self.nexchange)
        else:
            order = np.arange(self.nexchange)
        counter = 0
        while True:
            batch_indices = islice(order, batch_size*counter, batch_size*(counter+1))
            batch_indices = list(batch_indices)
            if len(batch_indices) == 0: return
            if len(batch_indices) < batch_size:
                nmissing = batch_size - len(batch_indices)
                batch_indices.extend(np.random.choice(self.nexchange, nmissing))
            batch_indices = np.array(batch_indices)
            if meta_tokens is None:
                yield counter, self.encode_inputs[batch_indices], self.decode_inputs[batch_indices], self.decode_targets[batch_indices]
            elif meta_tokens is 'most_common':
                original_lines =  batch_indices // 2 # // because 2 encodes per line
                batch_genres = [self.meta_reader.get_genre(line, most_common=True) for line in original_lines]
                #encoded_batch_genres = np.array([self.vocab.encode(bg) for bg in batch_genres])

                assert encoded_batch_genres.size == batch_size
                yield counter, self.encode_inputs[batch_indices], self.decode_inputs[batch_indices], self.decode_targets[batch_indices], batch_genres
            elif meta_tokens is 'all':
                original_lines =  batch_indices // 2 # // because 2 encodes per line
                encoded_batch_genres = np.zeros((batch_size, self.vocab.get_vocab_size()), dtype=int)
                for i, index in enumerate(batch_indices):
                    genres = self.meta_reader.get_genre(index)
                    encoded_genres = np.array([self.vocab.encode(x) for x in genres], dtype=int)
                    encoded_batch_genres[i, encoded_genres] = 1
                assert encoded_batch_genres.size == batch_size

                yield counter, self.encode_inputs[batch_indices], self.decode_inputs[batch_indices], self.decode_targets[bath_indices], encoded_batch_genres
               
            counter += 1

def load_default_objects(vocab_size=10000, sent_size=15):
    data_path = './data/Training_Shuffled_Dataset.txt'
    vocab = Vocab(); vocab.construct(data_path, vocab_size)
    dr = DataReader(vocab); dr.construct(data_path, sent_size=sent_size)
    return vocab, dr

def load_default_objects_with_meta_tokens(vocab_size=10000, sent_size=15):
    data_path = './data/Training_Shuffled_Dataset.txt'
    meta_path = './data/MetaInfo.txt'
    label_path = './data/Training_Shuffled_Dataset_Labels.txt'
    mr = MetaReader(meta_path, label_path)
    vocab = Vocab(); vocab.construct(data_path, vocab_size, mr)
    dr = DataReader(vocab, mr); dr.construct(data_path, sent_size=sent_size)
    assert vocab.get_vocab_size() == vocab_size
    return vocab, dr, mr

class SubmissionGenerator(object):
    def __init__(self, submission_folder):
        self.filename = os.path.join(submission_folder, 'group29.perplexity')

    def append_perplexities(self, perplexities):
        with open(self.filename, 'a') as file:
            for perplexity in perplexities[0]:
                file.write(str(perplexity)+'\n')
            file.close()
        print('Appended perplexities to ' + str(self.filename))

def self_test(path=None):
    vocab = Vocab()
    vocab.construct(path, 20000)
    data = DataReader(vocab)
    data.construct(path, 15)
    with open(path, 'r') as fin:
        first_line = fin.readline().strip().split('\t')
    print("Raw encode input: %s" %first_line[0])
    print("       Processed: %s" %" ".join([vocab.decode(x) for x in data.encode_data[0]]))
    print("")
    print("Raw decode input: %s" %first_line[1])
    print("       Processed: %s" %" ".join([vocab.decode(x) for x in data.decode_data[0]]))
    return



if __name__ == '__main__':
    path = sys.argv[1]
    self_test(path)
    


