# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# ATTENTION: The Code below is copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
# 
# ==============================================================================

import tensorflow as tf
from seq2seq import embedding_rnn_seq2seq
from seq2seq import sequence_loss_by_example
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
import numpy as np

class Seq2SeqModel(object):
    """Sequence-to-sequence model.

    This class implements a seq2seq RNN as described in this paper:
    https://arxiv.org/pdf/1409.3215.pdf

    The model consists of two multilayered LSTMs, one that maps input sequences
    to a fixed-length vector, and one that decodes the target sequence from the
    vector. Each LSTM consists of num_layers layers, each with
    1000 (<-- TODO: parametrize) cells. The model further uses 1000 (<-- TODO:
    parametrize) dimensional word embeddings.

    The model is designed to be a chatbot, meaning that the encoder-inputs and
    the decoder-inputs are of the same vocabulary.

    TODO: bucketing
    TODO: sampled softmax
    TODO: attention
    TODO: Option for forward_only
    """
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 max_sentence_length,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.
        Args:
          vocab_size: size of the vocabulary.
          hidden_size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          max_sentence_length: maximum sentence length that will be processed
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            learning_rate, trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_size)

        cell = single_cell()
        if num_layers > 1:
          cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_rnn_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size, #TODO: check if this makes sense
                output_projection=None,
                feed_previous=do_decode,
                dtype=dtype)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(max_sentence_length):
            self.encoder_inputs.append(tf.placeholder(tf.int32,
                shape=[None], name="encoder{0}".format(i)))
        for i in range(max_sentence_length + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,
                shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype,
                shape=[None], name="weight{0}".format(i)))
    
        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
            for i in range(len(self.decoder_inputs) - 1)]

        all_inputs = self.encoder_inputs + self.decoder_inputs + targets + self.target_weights
        with ops.name_scope(None, "model", all_inputs):
            with variable_scope.variable_scope(variable_scope.get_variable_scope()):
                self.outputs, _ = seq2seq_f(self.encoder_inputs,
                               self.decoder_inputs[:max_sentence_length], forward_only)
                self.losses = sequence_loss_by_example(self.outputs,
                        targets, self.target_weights[:max_sentence_length],
                        softmax_loss_function=None)

        params = tf.trainable_variables()
    
        if not forward_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
            self.update = opt.apply_gradients(
                  zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())


    def step(self, session, encoder_inputs, decoder_inputs,
        target_weights, forward_only):
        """Run a step of the model feeding the given inputs.
        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          forward_only: whether to do the backward step or only forward.
        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with max_sentence_length.
        """
        # Check if the sizes match.
        if len(encoder_inputs) != self.sentence_length:
            raise ValueError("Encoder length must be equal to max_sentence_length,"
                           " %d != %d." % (len(encoder_inputs), self.sentence_length))
        if len(decoder_inputs) != self.sentence_length:
            raise ValueError("Decoder length must be equal to max_sentence_length,"
                           " %d != %d." % (len(decoder_inputs), self.sentence_length))
        if len(target_weights) != self.sentence_length:
            raise ValueError("Weights length must be equal to max_sentence_length,"
                           " %d != %d." % (len(target_weights), self.sentence_length))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(self.sentence_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[self.sentence_length].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.losses]  # Loss for this batch.
        else:
            output_feed = [self.losses]  # Loss for this batch.
            for l in range(self.sentence_length):  # Output logits.
                output_feed.append(self.outputs[l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

