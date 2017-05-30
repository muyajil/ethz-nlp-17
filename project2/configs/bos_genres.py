# To capture genre style, when decoding use the genre tag (e.g. <comedy>) instead of <bos>
# TODO: read in meta info (utils.MetaReader)
# TODO: add genre tags to utils.Vocab
# TODO: extend utils.DataReader.get_iterator to interate over genre, adding genre tokens to feed_dict
# TODO: overwrite Model._get_bos_embedded to give tokens from feed_dict

import os
import sys
import tensorflow as tf

# safe (?) way to do sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from basic_seq2seq import Seq2Seq, LanguageSeq2Seq
