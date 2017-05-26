# Borrowed a lot from: https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb

import os
import sys
import numpy as np
import tensorflow as tf
import utils


class Config(object):
    vocab_size = 10
    embed_dim = encode_embed_dim = decode_embed_dim = 20
    encoder_hidden_units = decoder_hidden_units = 20
    batch_size = 5
    sequence_length = decode_sequence_length = encode_sequence_length = 10

class Seq2seq(object):
    def __init__(self, config):
        '''Basic Seq2seq implementation.

        Encoder inputs are fed through an RNN, final state is used to seed
        an RNN over decoder inputs. We will try augmenting the final state
        with outside information.

        This currently uses dynamic rnn cells, dynamic means sequence length can
        change from batch to batch, currently not taken advantage of.
        
        Inputs are tokens, which are embedded using a shared embedding table
        Loss is calculated on decoder outputs, with decoder_targets.

        self.encoder_inputs: tf.placeholder for encoder integer input tokens
            (encoder_sequence_length x batch_size)
        self.decoder_inputs: tf.placehodler for decoder integer input tokens
            (decoder_sequence_length x batch_size)
            decoder_inputs are fed into the decoder rnn, right now they are fixed
            -- i.e. it is difficult to generate
        self.decoder_targets: tf.placeholder for decoder integer target tokens
            (decoder_sequence_length x batch_size)
            loss is calculated with these tokens
            default: decoder_inputs shifted back one step.

        '''
        # Add encoder_inputs, decoder_inputs, decoder_targets
        self.config = config
        self.add_placeholders()
        self.encoder_inputs_embedded, self.decoder_inputs_embedded = \
            self.add_embedding(self.encoder_inputs, self.decoder_inputs)
        self.encoder_final_state = self.add_encoder_rnn(self.encoder_inputs_embedded)
        self.encoder_final_state_augmented = self.augment_final_state(self.encoder_final_state)
        self.decoder_outputs, self.decoder_final_state = \
            self.add_decoder_rnn(self.decoder_inputs_embedded, self.encoder_final_state_augmented)
        self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.config.vocab_size)
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits)
        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        return

    def add_placeholders(self):
        '''Adds the encode/decode input & decode label placeholders to the graph.
        '''
        # Inputs are sequence_length x batch_size !!
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(self.config.encode_sequence_length, self.config.batch_size),
            name='encoder_inputs')
        self.decoder_targets = tf.placeholder(dtype=tf.int32,
            shape=(self.config.decode_sequence_length, self.config.batch_size),
            name='decoder_targets')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(self.config.decode_sequence_length, self.config.batch_size),
            name='decoder_inputs')
        return

    def add_embedding(self, encoder_inputs, decoder_inputs):
        '''Adds embedding lookup on inputs using a shared embedding table.
        '''
        with tf.variable_scope('embedding'):
            embedding_shape = [self.config.vocab_size, self.config.embed_dim]
            self.embedding_table = tf.get_variable(name='embedding_table',
                shape=embedding_shape,
                initializer=tf.random_uniform_initializer(-1,1))
        encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_table, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_table, decoder_inputs)
        return encoder_inputs_embedded, decoder_inputs_embedded

    def add_encoder_rnn(self, encoder_inputs_embedded):
        '''Encoder rnn, used to access the hidden state.
        '''
        # Define encoder
        encoder_cell = tf.contrib.rnn.LSTMCell(self.config.encoder_hidden_units)
        _ , encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, encoder_inputs_embedded,
            dtype=tf.float32, time_major=True)
        return encoder_final_state

    def augment_final_state(self, encoder_final_state):
        '''Function to augment the encoder final state.
        '''
        return encoder_final_state

    def add_decoder_rnn(self, decoder_inputs_embedded, encoder_final_state):
        '''Decoder rnn, loss is calculated on its output.
        '''
        decoder_cell = tf.contrib.rnn.LSTMCell(self.config.decoder_hidden_units)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded,
            initial_state=encoder_final_state,
            dtype=tf.float32, time_major=True, scope='plain_decoder')
        return decoder_outputs, decoder_final_state

def memorize_test():
    config = Config()
    config.vocab_size = 10
    config.embed_dim = 20
    config.encoder_hidden_units = config.decoder_hidden_units = 20
    config.sequence_length = config.decode_sequence_length = config.encode_sequence_length = 10
    config.batch_size = 5

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Seq2seq(config)

        # Create a toy dataset
        ndata = 20 
        dataset = list()

        encoder_input_dataset = np.zeros((ndata, config.sequence_length), dtype=int)
        decoder_target_dataset = np.zeros_like(encoder_input_dataset)
        decoder_input_dataset = np.zeros_like(encoder_input_dataset)

        for i in range(ndata):
            size = np.random.randint(config.sequence_length - 2) + 1
            tokens = np.random.randint(config.vocab_size - 2, size=size) + 2
            assert 1 < min(tokens)
            encoder_input_dataset[i, :size] = tokens
            decoder_target_dataset[i, :(size + 1)] = np.append(tokens, 1)
            decoder_input_dataset[i, :(size + 1)] = np.append(1, tokens)

        sess.run(tf.global_variables_initializer())
        nbatches = 3001
        batches_in_epoch = 500
        loss_track = list()
        for batch in range(nbatches):
            batch_indices = np.random.randint(ndata, size=model.config.batch_size)
            encoder_inputs_ = encoder_input_dataset[batch_indices].T
            decoder_inputs_ = decoder_input_dataset[batch_indices].T
            decoder_targets_ = decoder_target_dataset[batch_indices].T
            feed_dict = {model.encoder_inputs: encoder_inputs_,
                         model.decoder_inputs: decoder_inputs_,
                         model.decoder_targets: decoder_targets_}
            _, batch_loss = sess.run([model.train_op, model.loss], feed_dict)
            loss_track.append(batch_loss)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(model.loss, feed_dict)))
                predict_ = sess.run(model.decoder_prediction, feed_dict)
                for i, (inp, pred) in enumerate(zip(feed_dict[model.encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(inp))
                    print('    predicted > {}'.format(pred))
                    if i >= 2: break
                    print()    
        assert loss_track[-1] < .05
if __name__ == '__main__':
    memorize_test()

