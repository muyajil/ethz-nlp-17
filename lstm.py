import time
import numpy as np
import tensorflow as tf
from model import LanguageModel
from utils import DataReader

class Config(object):
    """
    Holds model hyperparams and data information.
    """
    batch_size = 64
    state_size = 512
    embed_dim = 100
    vocab_size = 20000 + 2 # for UNK TODO
    sentence_length = 30
    data_path = "data/sentences.train"
    learning_rate = 0.5
    epochs = 1
    log_dir = "summaries"
    print_freq = 20

class Lstm(LanguageModel):

    def _get_variable(self, name, shape, weight_decay=None):
        '''
        The weight decay parameter gives the scaling constant for the
        L2-loss on the parameters.
        '''
        var = tf.get_variable(name, shape,
            initializer=tf.contrib.layers.xavier_initializer())

        if weight_decay is not None:
                wd = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
                tf.add_to_collection('losses', wd)

        return var


    def load_data(self):
        print("loading data..")
        data_reader = DataReader()
        data_reader.construct(self.config.data_path,
            self.config.vocab_size, self.config.sentence_length)
        return data_reader


    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int64,
                (self.config.batch_size, self.config.sentence_length))


    def create_feed_dict(self, input_batch):
        return {self.input_placeholder: input_batch}


    def add_embedding(self, input_data):
        """Add embedding layer that maps from vocabulary to vectors.
        Args:
            input_data: A tensor of shape (batch_size, sentence_length).
        Returns:
            wordvectors: A tensor of shape (batch_size, sentence_length, embed_dim)
        """
        with tf.variable_scope('embedding'):
            embedding = self._get_variable('embedding',
                [self.config.vocab_size, self.config.embed_dim])
        wordvectors = tf.nn.embedding_lookup(embedding, self.input_placeholder)

        return wordvectors
        

    def add_model(self, input_data):
        """Implements core of the model.
        Args:
            input_data: A tensor of shape (batch_size, sentence_length).
        Returns:
            sentence_logits: A tensor of shape (batch_size, sentence_length, vocab_size)
	     For each batch, there are the logits of for each word of the sentence
        """
        lstm = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.config.state_size)

        with tf.variable_scope('softmax_layer'):
            softmax_w = self._get_variable('softmax_w',[self.config.state_size, self.config.vocab_size])
            softmax_b = self._get_variable('softmax_b', [self.config.vocab_size])

        wordvectors = self.add_embedding(input_data)

        memory_state = tf.Variable(tf.zeros([self.config.batch_size, self.config.state_size]))
        hidden_state = tf.Variable(tf.zeros([self.config.batch_size, self.config.state_size]))
        state = (memory_state, hidden_state)
        sentence_logits = []

        with tf.variable_scope('model_state') as scope:
            for i in range(self.config.sentence_length-1):
                if i > 0:
                    scope.reuse_variables()

                x = wordvectors[:,i,:]
                output, state = lstm(x, state)
                logits = tf.matmul(output, softmax_w) + softmax_b
                sentence_logits.append(logits)

        return sentence_logits


    def add_perplexity_op(self, sentence_logits):
        # TODO is this redundant with add_loss_op? Maybe be incurring
        # computational overhead.
        """Adds ops for perplexity to the computational graph.

        Args:
            sentence_logits: A tensor of shape (batch_size, sentence_length, vocab_size)
        Returns:
            perplexity: 0-d tensor (scalar), 0-d tensor (scalar)
        """
        loss = 0.0 # in fact, has shape (batch_size,)
        for i in range(self.config.sentence_length-1):
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sentence_logits[i],
                labels=self.input_placeholder[:,i+1])

        perplexity = tf.pow(2.0, -loss)
        perplexity = tf.div(perplexity, self.config.sentence_length)
        mean_perplexity = tf.reduce_mean(perplexity)
        tf.summary.scalar("perplexity", mean_perplexity)
        return mean_perplexity


    def add_loss_op(self, sentence_logits):
        """Adds ops for loss to the computational graph.

        Args:
            sentence_logits: A tensor of shape (batch_size, sentence_length, vocab_size)
        Returns:
            loss, perplexity: 0-d tensor (scalar), 0-d tensor (scalar)
        """
        loss = 0.0
        for i in range(self.config.sentence_length-1):
            loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sentence_logits[i],
                labels=self.input_placeholder[:,i+1])

        loss = tf.div(loss, self.config.sentence_length)
        mean_loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", mean_loss)
        return mean_loss


    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


    def run_epoch(self, sess, input_data):
        """Runs an epoch of training.

        Trains the model for one-epoch.
    
        Args:
            sess: tf.Session() object
            input_data: utils.DataReader() object, with construct() already called
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        loss = 0.0
        for i, batch in input_data.get_iterator(self.config.batch_size):
            feed_dict = self.create_feed_dict(batch)
            # TODO kinda shitty that every time we add an op we have
            # to remember to put it in here. There should be some way
            # of automagically detecting which ops are available.
            _, loss_value, perplexity_value, merged_summary = sess.run([self.train_op, self.loss, self.perplexity, self.merged_summary_op], feed_dict=feed_dict)
            loss += loss_value
            self.summary_writer.add_summary(merged_summary, i)

            if i % self.config.print_freq == 0:
                msg = "\rbatch: %d loss: %.2f perplexity: %.2f" %(i, loss_value, perplexity_value)
                print(msg, end='')

        avg_loss = loss / (i+1)
        return avg_loss


    def fit(self, sess, input_data):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            input_data: utils.DataReader() object, with construct() already called
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        self.summary_writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())
        print("starting training..")
        for epoch in range(self.config.epochs):
            start_time = time.time()
            avg_loss = self.run_epoch(sess, input_data)
            # Note that the shuffle is done in get_iterator()
            duration = time.time() - start_time
            print('Epoch %d: loss = %.2f (%.3f sec)'
                         % (epoch, avg_loss, duration))
            losses.append(avg_loss)
        self.summary_writer.close()
        return losses


    def predict(self, sess, input_data):
        """Make predictions from the provided model.
        Args:
            sess: tf.Session()
            input_data: utils.DataReader() object, with construct() already called
        Returns:
            average_loss: Average loss of model.
            predictions: Predictions of model on input_data
        """
        # TODO: This whole function needs to be (re)done
        '''
        start_time = time.time()
        predictions = []
        
        for i, batch in input_data.get_iterator(self.config.batch_size):
            feed_dict = self.create_feed_dict(batch)
            # TODO: Maybe we cannot use the same "model" as in the learning,
            #             because there we always feed in the ground truth, and not the
            #             output word of the previous step.
            sentence_logits, loss_value = sess.run([self.sentence_logits, self.loss], feed_dict=feed_dict)
            for sentence in sentence_logits:
                
            predictions.append(self.to_words(sentence_logits))
            loss += loss_value
        avg_loss = loss / i

        duration = time.time() - start_time
        print('loss = %.2f (%.3f sec)' % (avg_loss, duration))
     '''
        pass


    def __init__(self, config):
        self.config = config
        self.learning_data = self.load_data()
        self.add_placeholders()
        self.sentence_logits = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.sentence_logits)
        self.perplexity = self.add_perplexity_op(self.sentence_logits)
        self.train_op = self.add_training_op(self.loss)
        self.merged_summary_op = tf.summary.merge_all()






