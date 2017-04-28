import tensorflow as tf
from lstm import Config
from lstm import Lstm

def test_LstmModel():
  config = Config()
  config.data_path = "data/sentences_debug.train"
  config.epochs = 2
  with tf.Graph().as_default():
    with tf.Session() as sess:
      model = Lstm(config)
      init = tf.global_variables_initializer()
      sess.run(init)
      losses = model.fit(sess, model.learning_data)

if __name__ == "__main__":
    test_LstmModel()
