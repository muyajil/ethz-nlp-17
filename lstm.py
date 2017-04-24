import tensorflow as tf
import preprocess
from itertools import islice
import numpy as np

state_size = 512
embed_dim = 100
vocab_size = 20000
sentence_length = 30
batch_size = 64

def get_fresh_state():
    memory_state = tf.zeros([batch_size, state_size])
    hidden_state = tf.zeros([batch_size, state_size])
    return (memory_state, hidden_state)
    
init = tf.global_variables_initializer()
lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
embedding = tf.get_variable('embedding', [vocab_size, embed_dim])
with tf.Session() as sess:
    sess.run(init)

    for n_batch, batch in preprocess.batches(batch_size):
        print('batch num: %d' % n_batch)
        wordvectors = tf.nn.embedding_lookup(embedding, batch)

        state = get_fresh_state()
        for i in range(sentence_length-1):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
                
            x = wordvectors[:,i,:]
            y = wordvectors[:,i+1,:]
            output, state = lstm(x, state)
