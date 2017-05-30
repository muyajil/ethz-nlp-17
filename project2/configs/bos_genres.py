# To capture genre style, when decoding use the genre tag (e.g. <comedy>) instead of <bos>
# TODO: read in meta info (utils.MetaReader)
# TODO: add genre tags to utils.Vocab
# TODO: extend utils.DataReader.get_iterator to interate over genre, adding genre tokens to feed_dict
# TODO: overwrite Model._get_bos_embedded to give tokens from feed_dict

import os
import sys
import tensorflow as tf

# _BASEDIR = ../  -- dir with models/, data/, basic_seq2seq
_BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# safe (?) way to do sys.path.append('../')
sys.path.append(_BASEDIR)
import utils
from basic_seq2seq import Seq2Seq, LanguageSeq2Seq

class BaselineConfig(object):
    vocab_size = 10000
    embed_dim = encode_embed_dim = decode_embed_dim = 500
    encoder_hidden_units = decoder_hidden_units = 100
    batch_size = 124 
    sequence_length = decoder_sequence_length = encoder_sequence_length = 20
    steps_per_checkpoint = 50
    max_epochs = 15
    gradient_clip_value = 200

    data_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset.txt')
    train_dir = os.path.join(_BASEDIR, 'models/baseline')


