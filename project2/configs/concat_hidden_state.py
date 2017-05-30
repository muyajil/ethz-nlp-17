# 1) Augment hidden state with one-hot encoding of most common genre
# 2) Augment hidden state with "multi"-hot encoding of genres

# TODO: overwrite Model.step to deal with the genre tag properly
# TODO: add sparse encoding of genres
# TODO: overwrite Model.augment_hidden_state, concat genre embeddings

import os
import sys
import tensorflow as tf
import numpy as np

# _BASEDIR = ../  -- dir with models/, data/, basic_seq2seq
_BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# safe (?) way to do sys.path.append('../')
sys.path.append(_BASEDIR)
import utils
from basic_seq2seq import Seq2Seq, LanguageSeq2Seq

class ConcatConfig(object):
    vocab_size = 10000
    embed_dim = 500
    encoder_hidden_units = 100
    batch_size = 124 
    sequence_length = decoder_sequence_length = encoder_sequence_length = 20
    steps_per_checkpoint = 50
    max_epochs = 15
    gradient_clip_value = 200

    data_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset.txt')
    meta_path = os.path.join(_BASEDIR, 'data/MetaInfo.txt')
    label_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset_Labels.txt')
    train_dir = os.path.join(_BASEDIR, 'models/genre_bos')

config = ConcatConfig()


