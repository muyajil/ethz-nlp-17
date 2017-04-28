import tensorflow as tf
import preprocess
from itertools import islice
import numpy as np

state_size = 512
embed_dim = 100
vocab_size = 20000
vocab_size+=2 # for UNK TODO
sentence_length = 30
batch_size = 64
epochs = 1

def _get_variable(name, shape, weight_decay=None):
    '''
    The weight decay parameter gives the scaling constant for the
    L2-loss on the parameters.
    '''
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    if weight_decay is not None:
        wd = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)

    return var

# TODO generalize by removing batch_size
def get_fresh_state():
    memory_state = tf.Variable(tf.zeros([batch_size, state_size]))
    hidden_state = tf.Variable(tf.zeros([batch_size, state_size]))
    return (memory_state, hidden_state)

def lstm_train(xinput):
    lstm = tf.contrib.rnn.BasicLSTMCell(state_size)

    with tf.variable_scope('foobar'): # TODO this is stupid
        softmax_w = _get_variable('softmax_w', [state_size, vocab_size])
        softmax_b = _get_variable('softmax_b', [vocab_size])
        embedding = _get_variable('embedding', [vocab_size, embed_dim])

    wordvectors = tf.nn.embedding_lookup(embedding, xinput)
    state = get_fresh_state()
    loss = 0.0
    for i in range(sentence_length-1):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        x = wordvectors[:,i,:]
        y = xinput[:,i+1]
        output, state = lstm(x, state)

        logits = tf.matmul(output, softmax_w) + softmax_b
        loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

    return tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
with tf.Session() as sess:
    xinput_placeholder = tf.placeholder(tf.int64, [None, sentence_length])
    lstm_loss = lstm_train(xinput_placeholder)
    train = optimizer.minimize(lstm_loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(epochs):
        # TODO shuffle between epochs?
        for n_batch, batch in preprocess.batches(batch_size):
            _, loss = sess.run([train, lstm_loss], feed_dict = {xinput_placeholder: np.array(batch)})

            print('batch num: %d\tloss: %.2f' % (n_batch, loss))

from utils import DataReader

class Config(object):
    state_size = 512
    embed_dim = 100
    vocab_size = 20000
    sentence_length = 30
    batch_size = 64
    epochs = 1
    corpus_path = sys.argv[1]
    dropout = 1

class LangRNN(LanguageModel):
    
    def __init__(self, config):
        self.config = config
        self.load_data(self.config.corpus_path)
        return

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.sentence_length])
        self.label_placeholder = tf.placehoder(tf.int32, shape=[None, self.config.sentence_length])
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        return

    def add_embedding(self):
        '''Adds embedding layer to the graph.

        Returns
            wordvectors: 3D tensor of shape (batch_size, sentence_length, embed_dim)
        '''
        Lshape = [len(self.dr.vocab), self.config.embed_dim]
        with tf.variable_scope("Embedding"):
            L = tf.get_variable("L", 
                shape=Lshape,
                initializer=tf.random_uniform_initializer(-1, 1),
                trainable=True)
        wordvectors = tf.nn.embedding_lookup(L, self.input_placeholder)
        return wordvectors 

    def add_model(self, inputs):
        '''Adds the recurrent layer to the graph.

        Args
            inputs: 3D tensor of shape (batch_size, sentence_length, embed_dim)
        Returns
            rnn_outputs: list of len sentence_size-1, elements 2D tensors of shape
                (batch_size, state_size)
        '''
        xinit = tf.contrib.layers.xavier_initializer()
        lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
        state = self._get_fresh_state()
        rnn_outputs = list()
        with tf.variable_scopr("lstm"):
            for indx in range(self.config.sentence_length - 1):
                if indx > 0: scope.reuse_variables()
                output, state = lstm(inputs[:, indx, :], state)
                rnn_outputs.append(output)
        return rnn_outputs 

    def add_projection(self, rnn_outputs):
        '''Adds the projection from hidden state to vocabulary.

        Args
            rnn_outputs: list of len (sentence_size - 1), elements 2D tensors of shape
                (batch_size, state_size)
        Returns
            outputs:  list of len (sentence_size - 1), elements 2D tensors of shape
                (batch_size, vocab_size)
        '''
        outputs = list()

        Ushape = (self.config.state_size, self.config.vocab_size)
        b2shape = (self.config.vocab_size, )
        xinit = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("Softmax"):
            U = tf.get_variable("U", shape=Ushape, initializer=xinit)
            b = tf.get_variable("b", shape=b2shape, initializer=xinit)
            for indx in range(self.config.sentence_length - 1):
                out = tf.matmul(rnn_outputs[indx], U) + b
                outputs.append(out)
        return outputs

    def add_loss_op(self, logits):
        '''Add loss calculation to graph.

        Args:
            logits:  list of len (sentence_size - 1), elements 2D tensors of shape
                (batch_size, vocab_size)
        Returns:
            loss: A 0-d tensor (scalar)
        '''
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=self.labels_placeholder)
        return tf.reduce_mean(loss)

    def load_data(self, path):
        self.data_reader = DataReader()
        self.data_reader.construct(path,
            self.config.vocab_size, self.config.sent_size)
        return
