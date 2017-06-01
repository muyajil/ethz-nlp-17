# Append embedding of most common genre to
#   each word embedding input to the decoder

# TODO: extend this and concat_one_hot_embeddings
#       to deal with multiple genres

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

class ConcatEmbeddingConfig(object):
    vocab_size = 10000

    # Size of genre embeddings
    genre_embed_dim = 10
    # Size of word embeddings
    embed_dim = 100
    # Size of RNN hidden states
    encoder_hidden_units = decoder_hidden_units = 100
    # Max sequence lengths
    sequence_length = decoder_sequence_length = encoder_sequence_length = 20

    # Optimization parameters
    batch_size = 124 
    max_epochs = 15
    gradient_clip_value = 200

    # Checkpoints
    steps_per_checkpoint = 50
    steps_per_validate = 50

    # Data paths
    data_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset.txt')
    label_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset_Labels.txt')
    valid_path = os.path.join(_BASEDIR, 'data/Validation_Shuffled_Dataset.txt')
    valid_label_path = os.path.join(_BASEDIR, 'data/Validation_Shuffled_Dataset_Labels.txt')
    meta_path = os.path.join(_BASEDIR, 'data/MetaInfo.txt')
   
    # Meta dirs
    train_dir = os.path.join(_BASEDIR, 'models/embed_concat')
    summary_dir = os.path.join(_BASEDIR, 'summaries/embed_concat')

config = ConcatEmbeddingConfig()

class GenreConcatEmbeddingSeq2Seq(LanguageSeq2Seq):
   
    def _load_data_objects(self, config):
        '''Load self.meta_reader, self.vocab, self.data_reader.

        Loads genre tags from meta_path, add them to Vocab before reading tokens.
        '''
        self.meta_reader = utils.MetaReader(config.meta_path, config.label_path)

        self.vocab = utils.Vocab()
        self.vocab.construct(config.data_path, config.vocab_size)

        self.data_reader = utils.DataReader(self.vocab, self.meta_reader)
        self.data_reader.construct(config.data_path, sent_size=config.sequence_length)

        self.valid_meta_reader = utils.MetaReader(config.meta_path, config.valid_label_path)
        self.valid_reader = utils.DataReader(self.vocab, self.valid_meta_reader)
        self.valid_reader.construct(config.valid_path, sent_size=config.sequence_length)

        all_genres = set()
        all_genres.update(self.meta_reader.genre_counter.keys())
        all_genres.update(self.valid_meta_reader.genre_counter.keys())
        self._map_genre_tkn_to_index = dict(zip(sorted(all_genres), range(len(all_genres))))

        config.num_genre = len(self._map_genre_tkn_to_index)
        assert self.vocab.get_vocab_size() == config.vocab_size
        return
        
    def add_placeholders(self):
        super(GenreConcatEmbeddingSeq2Seq, self).add_placeholders()
        self.genre_tags = tf.placeholder(dtype=tf.int32,
            shape=(self.config.batch_size),
            name='genre_tags')

        self.genre_embeddings = self._embed_genre()
        return

    def _embed_genre(self):
        with tf.variable_scope('Embedding'):
            embedding_shape = [self.config.num_genre, self.config.genre_embed_dim]
            self.genre_embedding_table = tf.get_variable(name='genre_embedding_table',
                shape=embedding_shape,
                initializer=tf.random_uniform_initializer(-1,1))
        return tf.nn.embedding_lookup(self.genre_embedding_table, self.genre_tags)

    def _get_bos_embedded(self):
        bos_embedded = super(GenreConcatEmbeddingSeq2Seq, self)._get_bos_embedded()
        return tf.concat([bos_embedded, self.genre_embeddings], axis=1)

    def _get_pad_embedded(self):
        pad_embedded = super(GenreConcatEmbeddingSeq2Seq, self)._get_pad_embedded()
        return tf.concat([pad_embedded, tf.zeros_like(self.genre_embeddings)], axis=1)

    def add_embedding(self, encoder_inputs, decoder_inputs):
        '''Adds embedding lookup on inputs using a shared embedding table.
        '''
        with tf.variable_scope('Embedding'):
            embedding_shape = [self.config.vocab_size, self.config.embed_dim]
            self.embedding_table = tf.get_variable(name='embedding_table',
                shape=embedding_shape,
                initializer=tf.random_uniform_initializer(-1,1))

        encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_table, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_table, decoder_inputs)

        genre_info_seq = tf.stack([self.genre_embeddings] * self.config.decoder_sequence_length)
        decoder_inputs_embedded = tf.concat([decoder_inputs_embedded, genre_info_seq], axis=2)
        return encoder_inputs_embedded, decoder_inputs_embedded

    def _loop_fn_get_next_input(self, previous_output):
        output_logits = tf.add(tf.matmul(previous_output, self.W), self.b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(self.embedding_table, prediction)
        next_input = tf.concat([next_input, self.genre_embeddings], axis=1)
        return next_input


    def get_batch_iter(self, reader):
        '''Iterator over inputs for self.step, for easy overwrite access.
        '''
        return reader.get_iterator(self.config.batch_size, meta_tokens='most_common')

    def construct_feed_dict(self, inputs):
        batch_id, encoder_inputs_, decoder_inputs_, decoder_targets_, genre_tokens_ = inputs
        genre_tags_ = np.array([self._map_genre_tkn_to_index[tkn] for tkn in genre_tokens_])

        # make feed_dict, run training & loss ops
        batch_start = utils.get_curr_time()
        feed_dict = {self.encoder_inputs: encoder_inputs_.T,
                     self.decoder_inputs: decoder_inputs_.T,
                     self.decoder_targets: decoder_targets_.T,
                     self.genre_tags: genre_tags_}
        return feed_dict

if __name__ == '__main__':
    config = ConcatEmbeddingConfig()
    tf.reset_default_graph()
    model = GenreConcatEmbeddingSeq2Seq(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)

