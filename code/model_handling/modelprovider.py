'''
Provide models ready to be trained using data from dataprovider.py
Build the complete tensorflow computation graph and return the tensorflow session
'''

import tensorflow as tf

def get_lstm_model():
    sess = tf.Session()
    return sess