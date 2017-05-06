import time
import numpy as np
import tensorflow as tf
import gensim

from model import LanguageModel
from utils import DataReader
_PROGRESS_BAR = False

class Config(object):
    """
    Holds model hyperparams and data information.
    """
    batch_size = 64
    state_size = 512
    embed_dim = 100
    vocab_size = 20000
    sentence_length = 30
    data_path = "data/sentences.train"
    learning_rate = 0.01
    epochs = 1
    log_dir = "summaries"
    print_freq = 10
    embed_path = None # pretrain: data/wordembeddings-dim100.word2vec
    submission_dir = "submissions"


def log(x, base=10):
    '''
    Computes log base `base` of the input `x`. There is no built-in

    Tensorflow function for this as of Mar 2016:
    https://github.com/tensorflow/tensorflow/issues/1666
    Temptation to monkey patch almost overwhelming =)
    '''
    base = float(base)
    return tf.log(x) / tf.log(base)


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

    def load_test_data(self):
        print("loading test data..")
        data_reader = DataReader()
        data_reader.construct('data/sentences.test',
            self.config.vocab_size, self.config.sentence_length)
        return data_reader

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32,
                (self.config.batch_size, self.config.sentence_length))


    def create_feed_dict(self, input_batch):
        return {self.input_placeholder: input_batch}


    def _load_pretrain_embedding(self, path):
        print("Loading external embeddings from %s" % path)
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        external_embedding = np.zeros(shape=(self.config.vocab_size, self.config.embed_dim))
        matches = 0
        for tok, idx in self.learning_data.vocab.word_to_index.items():
            if tok in model.vocab:
                external_embedding[idx] = model[tok]
                matches += 1
            else:
                print("%s not in embedding file" % tok)
                external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=self.config.embed_dim)
            
        print("%d words out of %d could be loaded" % (matches, self.config.vocab_size))
        return external_embedding
            

    def add_embedding(self, input_data):
        """Add embedding layer that maps from vocabulary to vectors.
        Args:
            input_data: A tensor of shape (batch_size, sentence_length).
        Returns:
            wordvectors: A tensor of shape (batch_size, sentence_length, embed_dim)
        """
        with tf.variable_scope('embedding'):
            embedding_shape = [self.config.vocab_size, self.config.embed_dim]
            if self.config.embed_path is None:
                embedding = self._get_variable('embedding', embedding_shape)
            else:
                pretrain = self._load_pretrain_embedding(self.config.embed_path)
                assert np.array_equal(pretrain.shape, embedding_shape)
                # TODO: When done this way, embeddings are stored more than once!
                #       This can lead to memory issues. The correct way is to adapt
                #       from the code of the TA but it is much more complicated..
                # See here:
                # http://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
                embedding = tf.get_variable('embedding',
                    shape=embedding_shape,
                    initializer=tf.constant_initializer(pretrain),
                    trainable=False)

        wordvectors = tf.nn.embedding_lookup(embedding, input_data)

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
            perplexity: A tensor of shape (batch_size, ) (One perplexity for each sentence)
        """
        sum_of_props = 0.0 # in fact, has shape (batch_size,) (checked)
        num_words_per_sentence = [self.config.sentence_length]*self.config.batch_size
        for i in range(self.config.sentence_length-1):
            ith_softmax = tf.nn.softmax(logits=sentence_logits[i]) # shape (batch_size, vocab_size) (checked)

            ith_probability = [] # will have shape (batch_size,) after the following loop (checked)
            
            for j in range(self.config.batch_size):
                prob_ij = ith_softmax[j, self.input_placeholder[j,i+1]] # is only one number (checked)
                if self.input_placeholder[j,i+1] == self.learning_data.vocab.word_to_index[self.learning_data.vocab.padding]:
                    prob_ij = 0.0
                    num_words_per_sentence[j]-=1
                ith_probability.append(prob_ij)

            sum_of_props += log(ith_probability, base=2)
        
        mean_log_prob_per_sentence = tf.div(sum_of_props, num_words_per_sentence) # shape (batch_size,)
        perplexity_per_sentence = tf.pow(2.0, -mean_log_prob_per_sentence) # shape (batch_size,)
        return perplexity_per_sentence


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
            _, loss_value, merged_summary = sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed_dict)
            loss += loss_value
            self.summary_writer.add_summary(merged_summary, i)

            if i % self.config.print_freq == 0:
                msg = "batch: %d loss: %.2f perplexity: %.2f" %(i, loss_value, perplexity_value)
                # TODO this constant whitespace is being recreated each time.
                if _PROGRESS_BAR:
                    print(' '*80, end='\r') # flush
                    print(msg, end='\r')
                else:
                    print(msg, end='\n')

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
        # TODO: proper embedding implementation
        # Need a sess.run call to assign pretrained embeddings to
        # self.embedding_placeholder
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
        self.test_data = self.load_test_data()
        self.add_placeholders()
        self.sentence_logits = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.sentence_logits)
        self.perplexities = self.add_perplexity_op(self.sentence_logits)
        self.train_op = self.add_training_op(self.loss)
        self.merged_summary_op = tf.summary.merge_all()


