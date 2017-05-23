import os
import sys

import numpy as np
from collections import Counter
from itertools import islice
import pickle

# TODO: Make modifying the encode/decoder preproc in data reader easy to modify

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

    def update(self, sentence):
        '''Update counts with words in sentence.
        '''
        if isinstance(sentence, str):
            self.word_counter.update(sentence.split())
        else:
            assert isinstance(sentence[0], str)
            self.word_counter.update(sentence)
        return

    def construct(self, path, vocab_size):
        '''Construct a vocabulary from path, cutting infrequent words to given size.
        '''

        print("Constructing vocabulary using:\n  Path: %s\n  Vocab size: %d" %(os.path.abspath(path), vocab_size))

        for line in open(path, 'r'):
            self.update(line)
        for key in [self.unknown, self.padding, self.begin, self.end]:
            self._insert(key)
        if vocab_size is None:
            words = [w for (w, _) in self.word_counter.most_common()]
        else:
            words = [w for (w, _) in self.word_counter.most_common(vocab_size - 4)]
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

class DataReader(object):
    def __init__(self, vocab=None):
        self.vocab = vocab
        if not vocab is None: self._store_tokens()
        self.encode_data = None
        self.decode_data = None
        self.nexchange = None
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
        self.encode_data = np.full((self.nexchange, sent_size), self._pad_token, dtype=int)
        self.decode_data = np.full((self.nexchange, sent_size), self._pad_token, dtype=int)
        print("Reading & tokenizing using:\n  Path: %s\n  Sentence size: %s" %(os.path.abspath(path), sent_size))

        for indx, line in enumerate(open(path, 'r')):
            a, b, c = self._tokenize_line(line)

            encode_first_tkns = self.preproc_encode_input(a, sent_size)
            encode_second_tkns = self.preproc_encode_input(b, sent_size)
            decode_first_tkns = self.preproc_decode_input(b, sent_size)
            decode_second_tkns = self.preproc_decode_input(c, sent_size)

            first_indx = 2 * indx
            second_indx = 2 * indx + 1

                # Encode has padding at start, Decode has padding at end
            self.encode_data[first_indx, -len(encode_first_tkns): ] = encode_first_tkns
            self.decode_data[first_indx, :len(decode_first_tkns)] = decode_first_tkns
            self.encode_data[second_indx, -len(encode_second_tkns):] = encode_second_tkns
            self.decode_data[second_indx, :len(decode_second_tkns)] = decode_second_tkns

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
        decode_tokens = [self._bos_token]
        decode_tokens.extend(tokens)
        decode_tokens = decode_tokens[:max_sent_size - 1]
        decode_tokens.append(self._eos_token)
        return decode_tokens

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
        assert self.encode_data is not None
        assert self.decode_data is not None
        perm = np.random.permutation(self.encode_data.shape[0])
        assert perm.size == len(self.encode_data) == len(self.decode_data)
        self.encode_data = self.encode_data[perm]
        self.decode_data = self.decode_data[perm]
        return

    def get_iterator(self, batch_size, shuffle=True):
        '''Iterator yielding batch number, batch_size sentences.
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
            yield counter, self.encode_data[batch_indices], self.decode_data[batch_indices]
            counter += 1

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
    


