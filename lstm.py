import time
import numpy as np
import tensorflow as tf
from model import LanguageModel
from utils import DataReader

class Config(object):
  """
  Holds model hyperparams and data information.
  """
  batch_size = 64
  state_size = 512
  embed_dim = 100
  vocab_size = 20000 + 2 # for UNK TODO
  sentence_length = 30
  data_path = "data/sentences.train"
  learning_rate = 0.5
  epochs = 1

class Lstm(LanguageModel):


  def _get_variable(self, name, shape, weight_decay=None):
    '''
    The weight decay parameter gives the scaling constant for the
    L2-loss on the parameters.
    '''
    var = tf.get_variable(name, shape,
      initializer=tf.contrib.layers.xavier_initializer())

    if weight_decay is not None:
        wd = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)

    return var


  def load_data(self):
    print "loading data.."
    data_reader = DataReader()
    data_reader.construct(self.config.data_path,
      self.config.vocab_size, self.config.sentence_length)
    return data_reader


  def add_placeholders(self):

    self.input_placeholder = tf.placeholder(tf.int64,
        (self.config.batch_size, self.config.sentence_length))


  def create_feed_dict(self, input_batch):

    return {self.input_placeholder: input_batch}


  def add_embedding(self, input_data):
    """Add embedding layer that maps from vocabulary to vectors.
    Args:
      input_data: A tensor of shape (batch_size, sentence_length).
    Returns:
      wordvectors: A tensor of shape (batch_size, sentence_length, embed_dim)
    """
    with tf.variable_scope('foobar'): # TODO this is stupid
      embedding = self._get_variable('embedding',
        [self.config.vocab_size, self.config.embed_dim])
    wordvectors = tf.nn.embedding_lookup(embedding, self.input_placeholder)

    return wordvectors
    

  def add_model(self, input_data):
    """Implements core of the model.
    Args:
      input_data: A tensor of shape (batch_size, sentence_length).
    Returns:
      logitss: A tensor of shape (batch_size, sentence_length, vocab_size)
	   For each batch, there are the logits of for each word of the sentence
    """
    lstm = tf.nn.rnn_cell.BasicLSTMCell(self.config.state_size)

    with tf.variable_scope('foobar'): # TODO this is stupid
        softmax_w = self._get_variable('softmax_w',
          [self.config.state_size, self.config.vocab_size])
        softmax_b = self._get_variable('softmax_b', [self.config.vocab_size])

    wordvectors = self.add_embedding(input_data)

    memory_state = tf.Variable(tf.zeros(
      [self.config.batch_size, self.config.state_size]))
    hidden_state = tf.Variable(tf.zeros(
      [self.config.batch_size, self.config.state_size]))
    state = (memory_state, hidden_state)
    logitss = []

    for i in range(self.config.sentence_length-1):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        x = wordvectors[:,i,:]
        output, state = lstm(x, state)
        logits = tf.matmul(output, softmax_w) + softmax_b
        logitss.append(logits)

    return logitss


  def add_loss_op(self, logitss):
    """Adds ops for loss to the computational graph.

    Args:
      logitss: A tensor of shape (batch_size, sentence_length, vocab_size)
    Returns:
      loss: A 0-d tensor (scalar) output
    """
    loss = 0.
    for i in range(self.config.sentence_length-1):
      loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logitss[i],
        labels=self.input_placeholder[:,i+1])
    loss = tf.div(loss, self.config.sentence_length)
    return tf.reduce_mean(loss)


  def add_training_op(self, loss):
    """Sets up the training Ops.
    Args:
      loss: Loss tensor.
    Returns:
      train_op: The Op for training.
    """
    optimizer = tf.train.AdamOptimizer(
      learning_rate=self.config.learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables))
    return train_op


  def run_epoch(self, sess, input_data):
    """Runs an epoch of training.

    Trains the model for one-epoch.
  
    Args:
      sess: tf.Session() object
      input_data: utils.DataReader() object, with construct() already called
    Returns:
      average_loss: scalar. Average minibatch loss of model on epoch.
    """
    loss = 0.
    for i, batch in input_data.get_iterator(self.config.batch_size):
      feed_dict = self.create_feed_dict(batch)
      _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
      loss += loss_value
    avg_loss = loss / (i+1)
    return avg_loss


  def fit(self, sess, input_data):
    """Fit model on provided data.

    Args:
      sess: tf.Session()
      input_data: utils.DataReader() object, with construct() already called
    Returns:
      losses: list of loss per epoch
    """
    losses = []
    print "starting training.."
    for epoch in range(self.config.epochs):
      start_time = time.time()
      avg_loss = self.run_epoch(sess, input_data)
      # Note that the shuffle is done in get_iterator()
      duration = time.time() - start_time
      print('Epoch %d: loss = %.2f (%.3f sec)'
             % (epoch, avg_loss, duration))
      losses.append(avg_loss)
    return losses


  def predict(self, sess, input_data):
    """Make predictions from the provided model.
    Args:
      sess: tf.Session()
      input_data: utils.DataReader() object, with construct() already called
    Returns:
      average_loss: Average loss of model.
      predictions: Predictions of model on input_data
    """
    # TODO: This whole function needs to be (re)done
    '''
    start_time = time.time()
    predictions = []
    
    for i, batch in input_data.get_iterator(self.config.batch_size):
      feed_dict = self.create_feed_dict(batch)
      # TODO: Maybe we cannot use the same "model" as in the learning,
      #       because there we always feed in the ground truth, and not the
      #       output word of the previous step.
      logitss, loss_value = sess.run([self.logitss, self.loss], feed_dict=feed_dict)
      for sentence in logitss:
        
      predictions.append(self.to_words(logitss))
      loss += loss_value
    avg_loss = loss / i

    duration = time.time() - start_time
    print('loss = %.2f (%.3f sec)' % (avg_loss, duration))
   '''
    pass


  def __init__(self, config):
    self.config = config
    self.learning_data = self.load_data()
    self.add_placeholders()
    self.logitss = self.add_model(self.input_placeholder)
    self.loss = self.add_loss_op(self.logitss)
    self.train_op = self.add_training_op(self.loss)






