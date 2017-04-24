import tensorflow as tf
import preprocess
from itertools import islice
import numpy as np

state_size = 10
embed_dim = 100
vocab_size = 20000
sentence_length = 30
batch_size = 42

memory_state = tf.zeros([batch_size, state_size])
hidden_state = tf.zeros([batch_size, state_size])
init = tf.global_variables_initializer()
lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
embedding = tf.get_variable('embedding', [vocab_size, embed_dim])
with tf.Session() as sess:
    sess.run(init)

    batch = list(islice(preprocess.iter(), batch_size))
    wordvectors = tf.nn.embedding_lookup(embedding, batch)

    v = wordvectors[:,0,:]
    print('input', v)

    output, state = lstm(v, (memory_state, hidden_state))
    print(output)
    print(state)
