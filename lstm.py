import tensorflow as tf

h_size = 10
vocab_size = 20000
sentence_length = 30

h_cell = tf.contrib.rnn.BasicRNNCell(h_size)

Wf = tf.get_variable('Wf', [10, h_size]) # TODO
Uf = tf.get_variable('Uf', [10, h_size])
bf = tf.get_variable('bf', [h_size])

def lstm_step(h_cell, x):
    ft = tf.dot(Wf, x) + tf.dot(Uf, h_cell) + bf
    

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer()
    
    sentence = tf.placeholder(tf.int64, shape=[sentence_length,])
    embedding = tf.get_variable('embedding', [vocab_size, h_size])
    word_vectors = tf.nn.embedding_lookup(embedding, sentence)
