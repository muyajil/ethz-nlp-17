import numpy as np
import tensorflow as tf
from model import LanguageModel

class Config(object):
  """
  Holds model hyperparams and data information.
  """
  batch_size = 64
  state_size = 512
  embed_dim = 100
  vocab_size = 20000 + 2 # for UNK TODO
  sentence_length = 30

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
    """
    Loads data from disk and stores it in memory.
    """
    #TODO: Should we call DataReader.construct here,
    # or is this supposed to be handled in the main.py file?
    # The main.py file, when calling Lstm(), could pass the iterator
    # from the DataReader to this method.

  def add_placeholders(self):
    """
    Adds placeholder variables to tensorflow computational graph.
    """
    self.input_placeholder = tf.placeholder(tf.int64,
        (self.config.batch_size, self.config.sentence_length))
    # TODO: tf.int64, because we represent each word with an id, right?

  def create_feed_dict(self, input_batch):
    """Creates the feed_dict for training the given step.
    Args:
      input_batch: A batch of input data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    return {self.input_placeholder: input_batch}

  def add_embedding(self, input_data):
    """Add embedding layer that maps from vocabulary to vectors.
    Args:
      input_data: A tensor of shape (batch_size, sentence_length).
    Returns:
      wordvectors: A tensor of shape (batch_size, sentence_length, embed_dim)
    """
    # TODO: Not sure if this is used the correct way, nor if
    # (batch_size, sentence_length, embed_dim) is the correct shape of wordvectors

    with tf.variable_scope('foobar'): # TODO this is stupid
      embedding = _get_variable('embedding',
        [self.config.vocab_size, self.config.embed_dim])
    wordvectors = tf.nn.embedding_lookup(embedding, input_placeholder)
    return wordvectors
    
  def add_model(self, input_data):
    """Implements core of the model.
    Args:
      input_data: A tensor of shape (batch_size, sentence_length).
    Returns:
      out: A tensor of shape (batch_size, TODO) <--- I'm confused, what exactly
						     is the output of the model?
						     Maybe just the last word that
						     is predicted? Or all the predicted
						     words?
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(state_size)

    with tf.variable_scope('foobar'): # TODO this is stupid
        softmax_w = _get_variable('softmax_w',
          [self.config.state_size, self.config.vocab_size])
        softmax_b = _get_variable('softmax_b', [self.config.vocab_size])

    wordvectors = self.add_embedding(input_data)

    memory_state = tf.Variable(tf.zeros([batch_size, state_size]))
    hidden_state = tf.Variable(tf.zeros([batch_size, state_size]))
    state = (memory_state, hidden_state)
    
    # TODO: Note that I slightly changed the code, such that the model
    # returns all the predicted words of the sentence
    # (Not sure if this is the way to go)
    outputs = []

    for i in range(self.config.sentence_length-1):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        x = wordvectors[:,i,:]
        y = input_data[:,i+1]
        output, state = lstm(x, state)
        outputs.append(output)

        # TODO:
        # Since we are using this Stanford template, we need to seperate the actual
        # model from the loss. This means that either this function needs to output
        # the logits, and the softmax is done in the loss op (note that we should use
        # tf.nn.sparse_softmax_cross_entropy_with_logits), or we find an alternative to
        # tf.nn.sparse_softmax_cross_entropy_with_logits that allows us to do the 
        # seperation  --> What should we do? 

        # logits = tf.matmul(output, softmax_w) + softmax_b
        # loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

    return outputs

  def add_loss_op(self, pred):
    """Adds ops for loss to the computational graph.

    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar) output
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def run_epoch(self, sess, input_data, input_labels):
    """Runs an epoch of training.

    Trains the model for one-epoch.
  
    Args:
      sess: tf.Session() object
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      average_loss: scalar. Average minibatch loss of model on epoch.
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def fit(self, sess, input_data, input_labels):
    """Fit model on provided data.

    Args:
      sess: tf.Session()
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      losses: list of loss per epoch
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def predict(self, sess, input_data, input_labels=None):
    """Make predictions from the provided model.
    Args:
      sess: tf.Session()
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      average_loss: Average loss of model.
      predictions: Predictions of model on input_data
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def __init__(self, config):
    self.config = config
    self.load_data()
    self.add_placeholders()
    self.pred = self.add_model(self.input_placeholder)
