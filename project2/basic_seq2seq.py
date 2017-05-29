# Borrowed a lot from: https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb

import os
import sys
import numpy as np
import tensorflow as tf
import utils
import seq2seq as google_code

class Config(object):
    vocab_size = 10
    embed_dim = encode_embed_dim = decode_embed_dim = 5
    encoder_hidden_units = decoder_hidden_units = 7
    batch_size = 5
    sequence_length = decode_sequence_length = encode_sequence_length = 10
    learning_rate = .0001

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
        # Parameters/Placeholders
        self.config = config
        self.add_placeholders()
        self._encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config.encoder_hidden_units)
        self._decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.config.decoder_hidden_units)

        # Encoder
        self.encoder_inputs_embedded, self.decoder_inputs_embedded = \
            self.add_embedding(self.encoder_inputs, self.decoder_inputs)
        self.encoder_final_state = self.add_encoder_rnn(self.encoder_inputs_embedded)

        # Overwrite augment_final_state
        self.encoder_final_state_augmented = self.augment_final_state(self.encoder_final_state)

        # Decoder
        self.decoder_outputs, self.decoder_final_state = \
            self.add_decoder_rnn(self.decoder_inputs_embedded, self.encoder_final_state_augmented)
        self.decoder_logits = self.project_onto_decoder_vocab(self.decoder_outputs)
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        # Training/Loss ops
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits)
        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)

        # op for generating sequences
        self.generated_preds, self.generated_logits = self.generate()
        self.generated_stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.generated_logits)
        self.generated_loss = tf.reduce_mean(self.generated_stepwise_cross_entropy)
        
        with tf.variable_scope('optimizer', reuse=None) as scope:
            self.adam = tf.train.AdamOptimizer()
            self.train_op = self.adam.minimize(self.loss)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            #self.gvs = self.optimizer.compute_gradients(self.stepwise_cross_entropy)
            #self.capped_gvs = [(tf.clip_by_value(grad, -15, 15), var) for grad, var in self.gvs]
            #self.train_op = self.optimizer.apply_gradients(self.capped_gvs)

        self.saver = tf.train.Saver(tf.global_variables())

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
        with tf.variable_scope('Embedding'):
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
        with tf.variable_scope('Encoder') as scope:
            _ , encoder_final_state = tf.nn.dynamic_rnn(
                self._encoder_cell, encoder_inputs_embedded,
                dtype=tf.float32, time_major=True)
        return encoder_final_state

    def augment_final_state(self, encoder_final_state):
        '''Function to augment the encoder final state.
        '''
        return encoder_final_state

    def add_decoder_rnn(self, decoder_inputs_embedded, encoder_final_state):
        '''Decoder rnn, loss is calculated on its output.
        '''
        with tf.variable_scope('Decoder') as scope:
            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                self._decoder_cell, decoder_inputs_embedded,
                initial_state=encoder_final_state,
                dtype=tf.float32, time_major=True)
        return decoder_outputs, decoder_final_state

    def project_onto_decoder_vocab(self, decoder_outputs):
        '''Calculates logits over vocabulary from the decoder outputs
        '''
        with tf.variable_scope("DecoderProjection") as scope:
            self.W = tf.get_variable(name='decode_weights',
                shape=[self.config.decoder_hidden_units, self.config.vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name='decode_bias',
                shape=[self.config.vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())

        # Need to flatten batch tensor for multiplication
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.config.vocab_size))
        return decoder_logits

    def _get_bos_embedded(self):
        bos_tokens = tf.stack(np.ones(self.config.batch_size, dtype=int))
        bos_embedding = tf.nn.embedding_lookup(self.embedding_table, bos_tokens)
        return bos_embedding

    def _get_pad_embedded(self):
        pad_tokens = tf.stack(np.zeros(self.config.batch_size, dtype=int))
        pad_embedding = tf.nn.embedding_lookup(self.embedding_table, pad_tokens)
        return pad_embedding

    def _loop_fn_initial(self):
        initial_elements_finished = (0 >= self.config.decode_sequence_length)  # all False at the initial step
        initial_input = self._get_bos_embedded()
        initial_cell_state = self.encoder_final_state_augmented
        initial_cell_output = None
        initial_loop_state = None  # we don't need to pass any additional information
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def _loop_fn_get_next_input(self, previous_output):
        output_logits = tf.add(tf.matmul(previous_output, self.W), self.b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(self.embedding_table, prediction)
        return next_input

    def _loop_fn_transition(self, time, previous_output, previous_state, previous_loop_state):
       
        elements_finished = (time >= self.config.decode_sequence_length) # this operation produces boolean tensor of [batch_size]
                                                      # defining if corresponding sequence has ended

        finished = tf.reduce_all(elements_finished) # -> boolean scalar
        #input = tf.gather(self.decoder_inputs_embedded, time)
        input = self._loop_fn_get_next_input(previous_output) 
        state = previous_state
        output = previous_output
        loop_state = None

        return (elements_finished, 
                input,
                state,
                output,
                loop_state)

    def _loop_fn(self, time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:    # time == 0
            assert previous_output is None and previous_state is None
            return self._loop_fn_initial()
        else:
            return self._loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    def generate(self):
        '''Generate output from the encoder final hidden state.
        
        This code and all the _loop_fn* functions are implemented from:
        https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
        '''
        with tf.variable_scope('Decoder', reuse=True):
            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self._decoder_cell, self._loop_fn)
        decoder_outputs = decoder_outputs_ta.stack() 

        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.config.vocab_size))
        decoder_prediction = tf.argmax(decoder_logits, 2)
        return decoder_prediction, decoder_logits

class LanguageSeq2Seq(Seq2seq):

    def _get_bos_embedded(self):
        bos_val = self.config.data_reader._bos_token 
        bos_tokens = tf.stack(np.full(self.config.batch_size, bos_val, dtype=int))
        bos_embedding = tf.nn.embedding_lookup(self.embedding_table, bos_tokens)
        return bos_embedding

    def _get_pad_embedded(self):
        pad_val = self.config.data_reader._pad_token
        pad_tokens = tf.stack(np.full(self.config.batch_size, pad_val,  dtype=int))
        pad_embedding = tf.nn.embedding_lookup(self.embedding_table, pad_tokens)
        return pad_embedding

def memorize_test():
    config = Config()
    config.vocab_size = 10
    config.embed_dim = 7
    config.encoder_hidden_units = config.decoder_hidden_units = 20
    config.sequence_length = config.decode_sequence_length = config.encode_sequence_length = 10
    config.batch_size = 5

    tf.reset_default_graph()

    model = Seq2seq(config)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

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

        if GENERATE:
            _, batch_loss = sess.run([model.generated_train_op, model.generated_loss], feed_dict)
        else:
            _, batch_loss = sess.run([model.train_op, model.loss], feed_dict)
        loss_track.append(batch_loss)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {} {}'.format(batch, '(Feeding previous output)' if GENERATE else '(Feeding decoder target)'))
            if GENERATE:
                print('  minibatch loss: {}'.format(sess.run(model.generated_loss, feed_dict)))
            else:
                print('  minibatch loss: {}'.format(sess.run(model.loss, feed_dict)))

            predict_, generate_ = sess.run([model.decoder_prediction, model.generated_preds], feed_dict)  
            for i, (inp, pred, gen) in enumerate(zip(feed_dict[model.encoder_inputs].T, predict_.T, generate_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    fed       > {}'.format(pred))
                print('    generated > {}'.format(gen))
                if i >= 5: break
                print()    
    return model                    




if __name__ == '__main__':
    GENERATE = False
    config = Config()
    config.vocab_size = 10
    config.embed_dim = 7
    config.encoder_hidden_units = config.decoder_hidden_units = 20
    config.sequence_length = config.decode_sequence_length = config.encode_sequence_length = 10
    config.batch_size = 5

    tf.reset_default_graph()

    model = Seq2seq(config)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

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

        if GENERATE:
            _, batch_loss = sess.run([model.generated_train_op, model.generated_loss], feed_dict)
        else:
            _, batch_loss = sess.run([model.train_op, model.loss], feed_dict)
        loss_track.append(batch_loss)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {} {}'.format(batch, '(Feeding previous output)' if GENERATE else '(Feeding decoder target)'))
            if GENERATE:
                print('  minibatch loss: {}'.format(sess.run(model.generated_loss, feed_dict)))
            else:
                print('  minibatch loss: {}'.format(sess.run(model.loss, feed_dict)))

            predict_, generate_ = sess.run([model.decoder_prediction, model.generated_preds], feed_dict)  
            for i, (inp, pred, gen) in enumerate(zip(feed_dict[model.encoder_inputs].T, predict_.T, generate_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    fed       > {}'.format(pred))
                print('    generated > {}'.format(gen))
                if i >= 5: break
                print()    
 
