import os
import sys
import tensorflow as tf

# safe (?) way to do sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basic_seq2seq import LanguageSeq2Seq as Model

class BaselineConfig(object):
    vocab_size = 10000
    embed_dim = encode_embed_dim = decode_embed_dim = 500
    encoder_hidden_units = decoder_hidden_units = 100
    batch_size = 124 
    sequence_length = decoder_sequence_length = encoder_sequence_length = 20
    steps_per_checkpoint = 50
    max_epochs = 15
    gradient_clip_value = 200

    data_path = './data/Training_Shuffled_Dataset.txt'
    train_dir = './models/baseline'

if __name__ == '__main__':
    config = BaselineConfig()
    tf.reset_default_graph()
    model = Model(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)

