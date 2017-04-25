'''
Provide models ready to be trained using data from dataprovider.py
Build the complete tensorflow computation graph and return the tensorflow session
'''

import tensorflow as tf

class NlpModel:
    
    session = None

    def __init__(self, 
        batch_size,
        seq_length,
        embedding_size,
        hidden_state_size,
        vocab_size):

        self.session = tf.Session()

        inputs = tf.placeholder(tf.float32, [batch_size, embedding_size, seq_length]) # word embeddings
        targets = tf.placeholder(tf.float32, [batch_size, embedding_size, seq_length]) # ground truth

        # TODO: xavier initializer
        softmax_W = tf.Variable(tf.zeros([hidden_state_size, vocab_size]))
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_state_size)

        old_state = cell.zero_state(batch_size, tf.float32)

        for i in range(seq_length):
            batch_outputs, batch_new_states = cell(inputs[:, i], batch_old_states) # how many outputs?
            batch_logits = tf.matmul(batch_outputs, softmax_W) # should work
            # TODO: Calculate loss
            batch_old_states = batch_new_states

        final_state = state
    
    def save_model():
        # TODO: save the trained model to disk
        return