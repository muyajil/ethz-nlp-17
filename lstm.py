import tensorflow as tf
import preprocess
from itertools import islice
import numpy as np

state_size = 10
embed_dim = 100
vocab_size = 20000
sentence_length = 30
batch_size = 1

# h_cell = tf.contrib.rnn.BasicRNNCell(state_size)

# Wf = tf.get_variable('Wf', [state_size, embed_dim])
# Uf = tf.get_variable('Uf', [42, state_size]) # TODO
# bf = tf.get_variable('bf', [state_size])

# def lstm(batch, embedding, H, Wf, Uf, bf):
#     wordvectors = tf.nn.embedding_lookup(embedding, batch)

#     for i in range(sentence_length-1):
#         curr_word = wordvectors[:,i-1,:]
#         ft = tf.matmul(Wf, tf.transpose(curr_word))
#         ft = tf.transpose(ft)
#         print(ft)
#         ft = tf.matmul(Wf, curr_word) + tf.matmul(Uf, tf.transpose(H)) + bf

# sentences = preprocess.iter()
    
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
    
#     embedding = tf.get_variable('embedding', [vocab_size, embed_dim])

#     batch = list(islice(sentences, batch_size))

#     lstm(batch, embedding, tf.get_variable('H', [batch_size, state_size]), Wf, Uf, bf)


init = tf.global_variables_initializer()
lstm = tf.contrib.rnn.BasicLSTMCell(state_size)
embedding = tf.get_variable('embedding', [vocab_size, embed_dim])

tf.placeholder()

c,h = lstm(v, state)

with tf.Session() as sess:
    sess.run(init)

    state = tf.zeros([batch_size, state_size])

    for batch in islice(preprocess.iter(), 10):

        batch = np.array([batch])

        wordvectors = tf.nn.embedding_lookup(embedding, batch)

        v = wordvectors[:,0,:]


