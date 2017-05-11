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
             dtype=tf.float32)
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
    """
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(hidden_size)

    cell = single_cell()
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        return seq2seq.embedding_rnn_seq2seq(
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

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        with variable_scope.variable_scope(variable_scope.get_variable_scope()):
            self.outputs, _ = seq2seq_f(encoder_inputs, decoder_inputs, False)
            self.losses = seq2seq.sequence_loss_by_example(outputs, targets, weights,
                softmax_loss_function=None)

    params = tf.trainable_variables()  
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses, params)
    clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                       max_gradient_norm)
    self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)
    self.saver = tf.train.Saver(tf.global_variables())




