import tensorflow as tf
import preprocess
from itertools import islice
import numpy as np

state_size = 512
embed_dim = 100
vocab_size = 20000
sentence_length = 30
batch_size = 64

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

    with tf.variable_scope('foobar'):
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

    # TODO mean(loss)
    return loss

with tf.Session() as sess:
    xinput_placeholder = tf.placeholder(tf.int64, [None, sentence_length])
    lstm_loss = lstm_train(xinput_placeholder)

    init = tf.global_variables_initializer()
    sess.run(init)


    for n_batch, batch in preprocess.batches(batch_size):
        print('batch num: %d' % n_batch)

        sess.run(lstm_loss, feed_dict = {xinput_placeholder: np.array(batch)})
