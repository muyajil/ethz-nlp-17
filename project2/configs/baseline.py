import os
import sys
import tensorflow as tf

# _BASEDIR = ../  -- dir with models/, data/, basic_seq2seq
_BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# safe (?) way to do sys.path.append('../')
sys.path.append(_BASEDIR)
import utils
from basic_seq2seq import LanguageSeq2Seq as Model


class BaselineConfig(object):
    vocab_size = 10000
    # Size of word embeddings
    embed_dim = 500
    # Size of RNN hidden states
    encoder_hidden_units = decoder_hidden_units = 100
    # Max sequence lengths
    sequence_length = decoder_sequence_length = encoder_sequence_length = 20

    # Optimization parameters
    batch_size = 124 
    max_epochs = 15
    gradient_clip_value = 200

    # Checkpoints
    steps_per_checkpoint = 100
    steps_per_validate = 500

    # Data Paths
    data_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset.txt')
    valid_path = os.path.join(_BASEDIR, 'data/Validation_Shuffled_Dataset.txt')

    # Meta Dirs
    train_dir = os.path.join(_BASEDIR, 'models/baseline')
    summary_dir = os.path.join(_BASEDIR, 'summaries/baseline')

if __name__ == '__main__':
    config = BaselineConfig()
    tf.reset_default_graph()
    model = Model(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)

