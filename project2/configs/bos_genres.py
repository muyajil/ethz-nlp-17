# To capture genre style, when decoding use the genre tag (e.g. <comedy>) instead of <bos>
# DONE: read in meta info (utils.MetaReader)
# DONE: add genre tags to utils.Vocab
# DONE: extend utils.DataReader.get_iterator to interate over genre, adding genre tokens to feed_dict
# TODO: overwrite Model._get_bos_embedded to give tokens from feed_dict
# TODO: overwrite Model.step to deal with the genre tag properly

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

class GenreBosConfig(object):
    vocab_size = 10000
    embed_dim = 500
    encoder_hidden_units = decoder_hidden_units = 100
    batch_size = 124 
    sequence_length = decoder_sequence_length = encoder_sequence_length = 20
    steps_per_checkpoint = 50
    max_epochs = 15
    gradient_clip_value = 200

    data_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset.txt')
    meta_path = os.path.join(_BASEDIR, 'data/MetaInfo.txt')
    label_path = os.path.join(_BASEDIR, 'data/Training_Shuffled_Dataset_Labels.txt')
    train_dir = os.path.join(_BASEDIR, 'models/genre_bos')

config = GenreBosConfig()


class GenreBosSeq2Seq(LanguageSeq2Seq):

    def _load_data_objects(self, config):
        '''Load self.meta_reader, self.vocab, self.data_reader.

        Loads genre tags from meta_path, add them to Vocab before reading tokens.
        '''
        self.meta_reader = utils.MetaReader(config.meta_path, config.label_path)
        self.vocab = utils.Vocab()
        self.vocab.construct(config.data_path, config.vocab_size, self.meta_reader)
        self.data_reader = utils.DataReader(self.vocab, self.meta_reader)
        self.data_reader.construct(config.data_path, sent_size=config.sequence_length)
        assert self.vocab.get_vocab_size() == config.vocab_size
        return

    def add_placeholders(self):
        super(GenreBosSeq2Seq, self).add_placeholders()
        self.genre_tags = tf.placeholder(dtype=tf.int32,
            shape=(self.config.batch_size),
            name='genre_tags')
        return

    def _get_bos_embedded(self):
        '''to seed the generator.
        '''
        bos_embedding = tf.nn.embedding_lookup(self.embedding_table, self.genre_tags)
        return bos_embedding

    def get_batch_iter(self):
        '''Iterator over inputs for self.step, for easy overwrite access.
        '''
        return self.data_reader.get_iterator(self.config.batch_size, meta_tokens='most_common')


    def step(self, sess, inputs, epoch=0):
        '''Run model for a single batch.
        '''
        if not hasattr(self, 'loss_track'): self.loss_track = list()
        if not hasattr(self.config, 'batches_per_epoch'):
            self.config.batches_per_epoch = ceil(self.data_reader.nexchange / self.config.batch_size)

        batch_id, encoder_inputs_, decoder_inputs_, decoder_targets_, genre_tags_ = inputs
        decoder_inputs_[:, 0] = genre_tags_

        # make feed_dict, run training & loss ops
        batch_start = utils.get_curr_time()
        feed_dict = {self.encoder_inputs: encoder_inputs_.T,
                     self.decoder_inputs: decoder_inputs_.T,
                     self.decoder_targets: decoder_targets_.T,
                     self.genre_tags: genre_tags_}
        _, batch_log_perp = sess.run([self.train_op, self.batch_log_perp_loss], feed_dict)
        batch_avg_log_perp = batch_log_perp.mean()
        batch_perplexity = np.exp(batch_avg_log_perp)
        self.loss_track.append(batch_perplexity)
        print(self._status(epoch, batch_id, batch_start), end='\r')

        if batch_id % self.config.steps_per_checkpoint == 0:
            self._checkpoint(sess, epoch, batch_id, batch_start, batch_perplexity, feed_dict)
        return

if __name__ == '__main__':
    tf.reset_default_graph()
    model = GenreBosSeq2Seq(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)

    if False:    
        meta_reader = utils.MetaReader(config.meta_path, config.label_path)
        vocab = utils.Vocab()
        vocab.construct(config.data_path, config.vocab_size, meta_reader)
        data_reader = utils.DataReader(vocab, meta_reader)
        data_reader.construct(config.data_path, sent_size=config.sequence_length)
        assert vocab.get_vocab_size() == config.vocab_size

