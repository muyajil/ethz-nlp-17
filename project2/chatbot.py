# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''

Running this program without --decode will start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint has a conversation.

'''

import numpy as np
import tensorflow as tf
from utils import DataReader
from model import Seq2SeqModel
import time
import math

#TODO: remove magic numbers
vocab_size = 40000
max_sentence_length = 50
data_path = "./data/Training_Shuffled_Dataset.txt"

# Model parameters
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", vocab_size, "Vocabulary size.")

# Data locations (TODO: adapt to our situation)
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")

# Run parameters
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = model.Seq2SeqModel(
        FLAGS.vocab_size,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        max_sentence_length,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():
    
    # TODO: Data preparation here
    from_train = None
    to_train = None
    from_dev = None
    to_dev = None

    data_reader = DataReader()
    data_reader.construct(data_path, vocab_size, max_sentence_length)
    
    with tf.Session() as sess:

        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            start_time = time.time()

        for (i, encoder_inputs, decoder_inputs) in data_reader.get_iterator(FLAGS.batch_size):

            start_time = time.time()
            target_weights = [1. * len(decoder_inputs)]
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint and print statistics.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                    "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "chatbot.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()


def decode():
    with tf.Session() as sess:

        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # TODO: Load vocabulary.
        data = DataReader()
        data_reader.construct(data_path, vocab_size, max_sentence_length)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # TODO: maybe remove BOS tag? (-> is not present on seq2seq tutorial)
            encoder_input = [data.vocab.encode(data.vocab.begin)]
            for i, word in enumerate(sentence.split()):
                if i == max_sentence_length-2: break
                encoder_input.append(data.vocab.encode(word))
            decoder_input = [data.vocab.begin] # Here should be the GO! tag
					   # I treat BOS tag as GO tag here
            target_weights = [0.]
            # TODO: padding and tagging for encoder_input 

            _, _, output_logits = model.step(sess, encoder_input, decoder_input,
                                       target_weights, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data.vocab.encode(data.vocab.end) in outputs:
                outputs = outputs[:outputs.index(data.vocab.encode(data.vocab.end))]

            print(" ".join([data.vocab.decode(output) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def self_test():
    with tf.Session() as sess:
        print("Self-test for the model.")
    
        # Create model with vocabularies of 10, 2 layers of 32.
        model = Seq2SeqModel(10, 32, 2, 5.0, 2, 0.3, 0.99, 2)
        sess.run(tf.global_variables_initializer())
        for _ in range(5):
            model.step(sess, [[1, 3, 5] , [1, 3, 3]],
                [[1, 3, 5] , [1, 3, 3]], [1., 1.], False)

def main(_):
    if FLAGS.self_test:
      self_test()
    elif FLAGS.decode:
      decode()
    else:
      train()

if __name__ == "__main__":
    tf.app.run()

