import os
import sys
import tensorflow as tf

def continue_sentence(sess, model, tokens, gen_size=20):
    '''Continue a sentence given list of tokens.
    
    Args:
        sess: tf.session
        model: language model, configured to generate sentences
        tokens: list of tokens that start the sentence
        gen_size: max size of continued sentence

    Returns:
        sentence: list of tokens (int) in full sentence
        perplexity: average perplexity of predicted words
    '''

    correct_config = model.config.batch_size = 1
        and model.config.sentence_length = 1
        and model.config.num_steps = 1
    assert correct_config, "Necessary: batch_size == sentence_length == num_steps == 1"

    state = model._get_initial_state().eval()
    p_w_next = None
    sentence = list()
    perplexity = 0
    n_generated_words = 0
    for i in range(gen_size):
        try:
            w_curr = tokens[i]
        except IndexError:
            assert not p_w_next is None
            w_curr = tf.argmax(p_w_next)
            # Increment perplexity if we predicted w_curr
            perplexity += w_next_perp
            n_generated_words += 1

        sentence.append(w_curr)
        if w_curr == stop_symbol: break
        feed_dict = {model.input_placeholder: [w_curr], model.state: state}

        # TODO: mayber verify this is correct
        ops = [model.sentence_logits[-1], model.state, model.perplexity_op]
        w_next_logits, state, w_next_perp = sess.run(ops, feed_dict=feed_dict)
        p_w_next = tf.softmax(w_next_logits)
        perplexity = perplexity / n_generated_words

    return sentence, perplexity

def run(sent_path, sess, gen_model, gen_size=20, append_bos=True):
    '''Generate sentences using gen_model and starts from sent_path.

    Args:
        sent_path (str): path to text file of sentence starts
        sess: tf.session
        gen_model: pretrained language model used to generate sentences
        gen_size (int): max length of generated sentences
        append_bos (bool): append the <bos> token to start of sentence

    Returns:
        sentence_list: list of strings of human readable completed sentences
        perplexity_list: list of perplexity values (one for each sentence)
    '''

    data_reader = gen_model.learning_data.data_reader
    vocab = gen_model.learning_data.vocab
    stop_symbol = vocab.encode(vocab.end)

    sentence_list = list()
    perplexity_list = list()
    for line in open(sent_path, 'r'):
        tokens = list()
        # add the <bos> symbol to every sentence we read
        if append_bos: tokens.append(vocab.encode(vocab.start))
        # parse start of sentence
        tokens.extend(data_reader.parse_line(line))
        assert tokens < gen_size, "Probably using the wrong file -- see email"

        # continue sentence
        tokens, perplexity = continue_sentence(sess, gen_model, tokens, gen_size=gen_size)
        # translate into human readable
        sentence_text = [vocab.decode(tkn) for tkn in tokens]

        # store for later
        sentence_list.append(sentence_text)
        perplexity_list.append(perplexity)
    return sentence_list, perplexity_list

def load_sess(path):
    '''Load the session object from a pretrained model.
    '''
    raise NotImplementedError

class Config(object):
    """
    Holds model hyperparams and data information.
    """
    batch_size = 64
    state_size = 512
    embed_dim = 100
    vocab_size = 20000
    sentence_length = 30
    num_steps = sentence_length - 1
    data_path = "data/sentences.train"
    test_path = "data/sentences.test"
    learning_rate = 0.01
    epochs = 1
    log_dir = "summaries"
    print_freq = 10
    embed_path = None # pretrain: data/wordembeddings-dim100.word2vec
    down_project = None
    submission_dir = "submissions"

if __name__ == '__main__':
    sess_path = sys.argv[1]
    sent_path = 'data/sentences.continuation'

    # I think most of the config gets ignored anyways...
    gen_config = Config()
    gen_config.batch_size = 1
    gen_config.sentence_length = 1
    gen_config.num_steps = 1

    with tf.Graph().as_default():
        # create generator model
        with tf.variable_scope('RNNLM_Generate') as scope:
            gen_model = Lstm(gen_config)

        # load session from pretrained model
        # use learned parameters to complete the sentences
        init = tf.global_variables_initializer()
        with load_sess(sess_path) as sess:
            sess.run(init)
            sentence_list = run(sent_path, sess, gen_model)
    # TODO: format the output and write to file

