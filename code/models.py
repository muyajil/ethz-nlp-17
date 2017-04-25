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
        vocab_size,
        summaries_dir):

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
            # TODO: Add loss and other information we want in tensorboard to tf.summary
            #       E.g. tf.summary.scalar(loss)
            batch_old_states = batch_new_states

        merged_summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(summaries_dir, self.session.graph)
        # TODO: I don't yet get how to exactly export the summaries, according to the tutorial, session.run() returns the summary
        #       And then we need to call the writer to serialize the summary to disk
        #       But after that it's easy, just launch tensorboard and point it to the summaries_dir
        final_state = state
    
    def save_model():
        # TODO: save the trained model to disk
        return