import tensorflow as tf
import argparse
from lstm import Config
from lstm import Lstm
import time

PARSER = argparse.ArgumentParser() 

def main(config):
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            model = Lstm(config)
            init = tf.global_variables_initializer()
            sess.run(init)
            losses = model.fit(sess, model.learning_data)
            saver = tf.train.Saver()
            saver.save(sess, 'models/rnn-language-model'+ str(time.time()))

if __name__ == "__main__":
    PARSER.add_argument("--predef", help="Predefined mode for all arguments", action='store_true')
    PARSER.add_argument("--embed_path", help="Load word embeddings from path", type=str)
    PARSER.add_argument("--data_path", help="Training Data file", type=str)
    PARSER.add_argument("--summaries_dir", help="Directory where the summaries for Tensorboard will be stored", type=str)
    PARSER.add_argument("--batch_size", help="Batch size for the RNN", type=int)
    PARSER.add_argument("--seq_length", help="Sequence Length of the RNN", type=int)
    PARSER.add_argument("--embedding_size", help="Embedding Size of the words", type=int)
    PARSER.add_argument("--hidden_state", help="The size of the hidden states of the RNN", type=int)
    PARSER.add_argument("--vocab_size", help="The size of your vocabulary", type=int)
    PARSER.add_argument("--learning_rate", help="Learning Rate for AdamOptimizer", type=float)
    PARSER.add_argument("--epochs", help="How many training epochs", type=int)
    args = PARSER.parse_args()

    config = Config()
    if args.predef:
        main(config)
    else:
        config.data_path = args.data_path
        config.batch_size = args.batch_size
        config.sentence_length = args.seq_length
        config.embed_dim = args.embedding_size
        config.state_size = args.hidden_state
        config.vocab_size = args.vocab_size
        config.learning_rate = args.learning_rate
        config.epochs = args.epochs
        config.log_dir = args.summaries_dir
        config.embed_path = args.embed_path
