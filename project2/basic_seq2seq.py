# Borrowed a lot from: https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb

# TODO: add option to load pretrained word embeddings,
#   e.g. self.embeddings = load_embeddings(pretrain_path, vocab)
#       which overwrites the intialization with embeddings for words
#       that exist in both vocabularies. Keep the embeddings as trainable

import os
import sys
import numpy as np
import tensorflow as tf
import utils
from seq2seq import sequence_loss_by_example
from math import ceil

class Config(object):
    vocab_size = 10
    embed_dim = 5
    encoder_hidden_units = decoder_hidden_units = 7
    batch_size = 5
    sequence_length = decoder_sequence_length = encoder_sequence_length = 10
    learning_rate = .0001

class Seq2Seq(object):
    def __init__(self, config):
        '''Basic Seq2Seq implementation.

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
        self._handle_config()
        self.add_placeholders()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        try:
            self.weights = tf.cast(tf.not_equal(self.decoder_targets, self.config.pad_symbol), tf.float32)
        except AttributeError:
            self.weights = tf.ones(self.decoder_targets.get_shape(), dtype=tf.float32)
            
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
        # shape: decoder_sequence_length x batch_size x decoder_vocab
        self.decoder_logits = self.project_onto_decoder_vocab(self.decoder_outputs)
        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        # Training/Loss ops
        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits)
        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.batch_log_perp_loss = sequence_loss_by_example(
            logits=tf.unstack(self.decoder_logits),   # list of 2D tensors (batch_size x vocab_size), len=decoder_sequence_length
            targets=tf.unstack(self.decoder_targets), # list of 1D tensors (batch_size), len=decoder_sequence_length
            weights=tf.unstack(self.weights),         # list of 1D tensors (batch_size), len=decoder_sequence_length
            average_across_timesteps=False)
        self.weighted_average_batch_log_perp_loss = tf.divide(tf.reduce_sum(self.batch_log_perp_loss), tf.reduce_sum(self.weights))
        self.perp_fed = tf.exp(self.weighted_average_batch_log_perp_loss)
        self.perp_summary_fed = tf.summary.scalar('perp_fed', self.perp_fed)

        # op for generating sequences
        self.generated_preds, self.generated_logits = self.generate()
        self.generated_stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.config.vocab_size, dtype=tf.float32),
            logits=self.generated_logits)
        self.generated_batch_log_perp_loss = sequence_loss_by_example(
            logits=tf.unstack(self.generated_logits),   # list of 2D tensors (batch_size x vocab_size), len=decoder_sequence_length
            targets=tf.unstack(self.decoder_targets),   # list of 1D tensors (batch_size), len=decoder_sequence_length
            weights=tf.unstack(self.weights),           # list of 1D tensors (batch_size), len=decoder_sequence_length
            average_across_timesteps=False)
        self.generated_weighted_average_batch_log_perp_loss = \
            tf.divide(tf.reduce_sum(self.generated_batch_log_perp_loss), tf.reduce_sum(self.weights))

        self.perp_gen = tf.exp(self.generated_weighted_average_batch_log_perp_loss)
        self.perp_summary_gen = tf.summary.scalar('perp_gen', self.perp_gen)

        # Ok to use adam and gradient clipping. These guys used lr=.001, clip at 200 (!!)
        # https://arxiv.org/pdf/1511.08400v7.pdf
        with tf.variable_scope('optimizer', reuse=None) as scope:
            if (not hasattr(self.config, 'gradient_clip_value')) or self.config.gradient_clip_value is None:
                self.adam = tf.train.AdamOptimizer()
                self.train_op = self.adam.minimize(self.loss, global_step=self.global_step)
            else:
                clip = self.config.gradient_clip_value
                self.optimizer = tf.train.AdamOptimizer()
                self.gvs = self.optimizer.compute_gradients(self.stepwise_cross_entropy)
                self.capped_gvs = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in self.gvs]
                self.train_op = self.optimizer.apply_gradients(self.capped_gvs, global_step=self.global_step)

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        return

    def _handle_config(self):
        '''Asserts, summary printing.
        '''
        if not os.path.exists(self.config.train_dir):
            os.makedirs(self.config.train_dir)

        if not os.path.exists(self.config.summary_dir):
            os.makedirs(self.config.summary_dir)

        assert os.path.exists(self.config.data_path)

    def add_placeholders(self):
        '''Adds the encode/decode input & decode label placeholders to the graph.
        '''
        # Inputs are sequence_length x batch_size !!
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(self.config.encoder_sequence_length, self.config.batch_size),
            name='encoder_inputs')
        self.decoder_targets = tf.placeholder(dtype=tf.int32,
            shape=(self.config.decoder_sequence_length, self.config.batch_size),
            name='decoder_targets')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32,
            shape=(self.config.decoder_sequence_length, self.config.batch_size),
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

    def project_onto_decoder_vocab(self, decoder_outputs, reuse=None):
        '''Calculates logits over vocabulary from the decoder outputs
        '''
        with tf.variable_scope("DecoderProjection", reuse=reuse) as scope:
            self.W = tf.get_variable(name='decode_weights',
                shape=[self.config.decoder_hidden_units, self.config.vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name='decode_bias',
                shape=[self.config.vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())

        # Need to flatten batch tensor for multiplication
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        tf.assert_equal(decoder_max_steps, tf.constant(self.config.decoder_sequence_length))
        tf.assert_equal(decoder_batch_size, tf.constant(self.config.batch_size))
        tf.assert_equal(decoder_dim, tf.constant(self.config.decoder_hidden_units))

        # Need to do this or else tensorflow forgets the shape of decoder_logits
        decoder_max_steps = self.config.decoder_sequence_length
        decoder_batch_size = self.config.batch_size
        decoder_dim = self.config.decoder_hidden_units

        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.config.vocab_size))
        return decoder_logits

    def _get_bos_embedded(self):
        bos_tokens = tf.stack(np.ones(self.config.batch_size, dtype=int))
        bos_embedded = tf.nn.embedding_lookup(self.embedding_table, bos_embedded)
        return bos_embedding

    def _get_pad_embedded(self):
        pad_tokens = tf.stack(np.zeros(self.config.batch_size, dtype=int))
        pad_embedded = tf.nn.embedding_lookup(self.embedding_table, pad_embedded)
        return pad_embedding

    def generate(self):
        '''Generate output from the encoder final hidden state.
        
        This code and all the _loop_fn* functions are implemented from:
        https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
        '''
        with tf.variable_scope('Decoder', reuse=True):
            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self._decoder_cell, self._loop_fn)
        decoder_outputs = decoder_outputs_ta.stack() 
        decoder_logits = self.project_onto_decoder_vocab(decoder_outputs, reuse=True)
        decoder_prediction = tf.argmax(decoder_logits, 2)
        return decoder_prediction, decoder_logits

    # Following functions are from the tutorial linked above.

    def _loop_fn_initial(self):
        initial_elements_finished = (0 >= self.config.decoder_sequence_length)  # all False at the initial step
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
       
        elements_finished = (time >= self.config.decoder_sequence_length) # this operation produces boolean tensor of [batch_size]
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


class LanguageSeq2Seq(Seq2Seq):
    '''Seq2Seq operations that require config.data_reader.
    '''

    def __init__(self, config):
        self._load_data_objects(config)
        config.pad_symbol = self.data_reader._pad_token
        config.batches_per_epoch = ceil(self.data_reader.nexchange / config.batch_size)
        super(LanguageSeq2Seq, self).__init__(config)
        return

    def _load_data_objects(self, config):
        self.vocab = utils.Vocab()
        self.vocab.construct(config.data_path, config.vocab_size)
        self.data_reader = utils.DataReader(self.vocab)
        self.data_reader.construct(config.data_path, sent_size=config.sequence_length)
        if hasattr(config, 'valid_path'):
            self.valid_reader = utils.DataReader(self.vocab)
            self.valid_reader.construct(config.valid_path, sent_size=config.sequence_length)
        else:
            self.valid_reader = None
        return

    def _get_bos_embedded(self):
        '''to seed the generator.
        '''
        bos_val = self.data_reader._bos_token 
        bos_tokens = tf.stack(np.full(self.config.batch_size, bos_val, dtype=int))
        bos_embedding = tf.nn.embedding_lookup(self.embedding_table, bos_tokens)
        return bos_embedding

    def _get_pad_embedded(self):
        '''To pad the generator.
        '''
        pad_val = self.data_reader._pad_token
        pad_tokens = tf.stack(np.full(self.config.batch_size, pad_val,  dtype=int))
        pad_embedding = tf.nn.embedding_lookup(self.embedding_table, pad_tokens)
        return pad_embedding

    def train(self, sess):
        '''Run model for all epochs
        '''
        print()
        print()
        print("Starting Training !!")
        print()

        self.loss_track = list()
        self.train_start_time = utils.get_curr_time()
        self.batches_per_epoch = ceil(self.data_reader.nexchange / self.config.batch_size)

        if hasattr(self.config, 'summary_dir'):
            train_summary_dir = os.path.join(self.config.summary_dir, 'train')
            valid_summary_dir = os.path.join(self.config.summary_dir, 'valid')
            if not os.path.exists(train_summary_dir):
                os.makedirs(train_summary_dir)
            else:
                print("WARNING: training summary dir already exists ({})".format(train_summary_dir))
            if not os.path.exists(valid_summary_dir):
                os.makedirs(valid_summary_dir)
            else:
                print("WARNING: validation summary dir already exists ({})".format(valid_summary_dir))
            self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            self.valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
            print("TENSORBOARD logdir: {}".format(self.config.summary_dir))
        else:
            print("WARNING: No tensorboard support for this run.")
            self.summary_writer = None
        for epoch_id in range(self.config.max_epochs):
            self.run_epoch(sess, epoch_id)
        return

    def get_batch_iter(self, reader):
        '''Iterator over inputs for self.step, for easy overwrite access.
        '''
        return reader.get_iterator(self.config.batch_size, meta_tokens=None)

    def construct_feed_dict(self, inputs):
        batch_id, encoder_inputs_, decoder_inputs_, decoder_targets_ = inputs
        feed_dict = {self.encoder_inputs: encoder_inputs_.T,
                     self.decoder_inputs: decoder_inputs_.T,
                     self.decoder_targets: decoder_targets_.T}
        return feed_dict
        

    def run_epoch(self, sess, epoch_id=0):
        '''Run model for a single epoch.
        '''
        epoch_start = utils.get_curr_time()
        batch_iter = self.get_batch_iter(self.data_reader)
        for inputs in batch_iter:
            self.step(sess, inputs, epoch_id)

        checkpoint_path = os.path.join(model.config.train_dir, 'chatbot_epoch_%d.ckpt'%(epoch_id + 1))
        self.saver.save(sess, checkpoint_path)
        print('Total Epoch time: {}'.format(utils.estimate_time(epoch_start)))
        return

    def step(self, sess, inputs, epoch=0):
        '''Run model for a single batch.
        '''
        if not hasattr(self, 'loss_track'): self.loss_track = list()
        if not hasattr(self.config, 'batches_per_epoch'):
            self.config.batches_per_epoch = ceil(self.data_reader.nexchange / self.config.batch_size)
        batch_id = inputs[0]
        # make feed_dict, run training & loss ops
        feed_dict = self.construct_feed_dict(inputs)
        batch_start = utils.get_curr_time()
        loss_op = self.weighted_average_batch_log_perp_loss
        #_, batch_log_perp, summary = sess.run([self.train_op, loss_op, self.tb_summary], feed_dict)
        _, batch_log_perp = sess.run([self.train_op, loss_op], feed_dict)
        batch_perplexity = np.exp(batch_log_perp)
        self.loss_track.append(batch_perplexity)
        print(self._status(epoch, batch_id, batch_start), end='\r')

        if batch_id % self.config.steps_per_checkpoint == 0:
            self._checkpoint(sess, epoch, batch_id, batch_start, batch_perplexity, feed_dict)
        if (not self.valid_reader is None) and batch_id % self.config.steps_per_validate == 0:
            self.validate(sess, epoch, batch_id)
        return

    def validate(self, sess, epoch, batch_id):
        '''Get average loss over whole validation set.
        '''
        valid_iter = self.get_batch_iter(self.valid_reader)
        inputs = next(valid_iter)
        feed_dict = self.construct_feed_dict(inputs)
        fed_loss_op = self.weighted_average_batch_log_perp_loss
        gen_loss_op = self.generated_weighted_average_batch_log_perp_loss 
        fed_loss, gen_loss = sess.run([fed_loss_op, gen_loss_op], feed_dict)
        status = self._status(epoch, batch_id) + ' Validating'

        global_step_, summary = sess.run([self.global_step, self.summary_op], feed_dict)
        self.valid_summary_writer.add_summary(summary, global_step_)
       
        print()
        print()
        print(status)
        print('  Perplexity (using fed logits): {}'.format(np.exp(fed_loss)))
        print('  Perplexity (using gen logits): {}'.format(np.exp(gen_loss)))
        print()
        print()

        return

    def _status(self, epoch, batch, batch_start=None):
        '''Prints a status line.
        '''
        status = 'Status: Epoch {} batch {}'.format(epoch, batch)
        if not batch_start is None:
            estimated_epoch_time = utils.estimate_time(batch_start, multiplier=self.config.batches_per_epoch)
            run_time = utils.estimate_time(self.train_start_time)
            status = status + ' (Total run time: {} Estimated epoch time: {})'.format(run_time, estimated_epoch_time)
        return status

    def _checkpoint(self, sess, epoch, batch_id, batch_start, batch_perplexity, feed_dict):
        '''Executes checkpoint saving, printing.
        '''
        print(self._status(epoch, batch_id, batch_start))
        print('  minibatch averge perplexity: {}'.format(batch_perplexity))
        predict_, generate_ = sess.run([self.decoder_prediction, self.generated_preds], feed_dict)  
        seqs = zip(feed_dict[self.encoder_inputs].T, feed_dict[self.decoder_targets].T, feed_dict[self.decoder_inputs].T, predict_.T, generate_.T)
        for i, (inp, tar, dinp, pred, gen) in enumerate(seqs):
            inp_decode = " ".join([self.vocab.decode(x) for x in inp[::-1]])
            tar_decode = " ".join([self.vocab.decode(x) for x in tar])
            dinp_decode = " ".join([self.vocab.decode(x) for x in dinp])
            pred_decode = " ".join([self.vocab.decode(x) for x in pred])
            gen_decode = " ".join([self.vocab.decode(x) for x in gen])
            print('  sample {}:'.format(i + 1))
            print('    input          > {}'.format(inp_decode))
            print('    target         > {}'.format(tar_decode))
            print('    (decode) input > {}'.format(dinp_decode))
            print('    fed            > {}'.format(pred_decode))
            print('    generated      > {}'.format(gen_decode))
            if i >= 9: break
            print()
        print()

        global_step_, summary = sess.run([self.global_step, self.summary_op], feed_dict)
        self.train_summary_writer.add_summary(summary, global_step_)
 
        # Save checkpoint
        checkpoint_path = os.path.join(self.config.train_dir, "chatbot.ckpt")
        self.saver.save(sess, checkpoint_path) 
        return


def memorize_test():
    config = Config()
    config.vocab_size = 10
    config.embed_dim = 7
    config.encoder_hidden_units = config.decoder_hidden_units = 20
    config.sequence_length = config.decoder_sequence_length = config.encoder_sequence_length = 10
    config.batch_size = 5

    tf.reset_default_graph()

    model = Seq2Seq(config)
    
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
    config.sequence_length = config.decoder_sequence_length = config.encoder_sequence_length = 9
    config.batch_size = 5
    config.pad_symbol = 0

    tf.reset_default_graph()

    model = Seq2Seq(config)
    
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

