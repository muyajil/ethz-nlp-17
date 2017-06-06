import os
import sys
import tensorflow as tf
import utils
import numpy as np

from configs import bos_genres, baseline, concat_embedding, concat_one_hot_word_embedding
_bdir = os.path.dirname(os.path.abspath(__file__))


DEBUG = False
if DEBUG:
    TEST_PATH = os.path.join(_bdir, 'data', 'Validation_Shuffled_Dataset.txt')
else:
    TEST_PATH = sys.argv[1] # path with triplets (ABC turns)

# TODO: choose the best model first (see eval.py for syntax)
# SESS_PATH must agree with Model !
SESS_PATH = os.path.join(_bdir, 'models', 'genre_bos', 'chatbot.ckpt')
config = bos_genres.config
Model = bos_genres.GenreBosSeq2Seq

# kind of shitty that we need to load the meta reader just to get num_genres ...
num_genres = utils.MetaReader(config.meta_path, config.label_path).num_genres
config.batch_size = num_genres 
model = Model(config)

# We want to get a perplexity for each pair for each genre i 
# and report the minimum perplexity across genres
# e.g. min_i {perp(a -> b | i) + perp(b -> c | i) }
# This is totally cheating -- we should predict the genre
# before we look at the score -- so maybe we should say
# something about this in the write up 

#TODO: need to write this information to file ... 
# genre_tags_ looks like [a, a, b, b, c, c, ...]
# so that we can run each pair of turns with the same genre

#genre_tokens_ = np.repeat(sorted(model.meta_reader.genre_set), 2)
genre_tokens_ = sorted(model.meta_reader.genre_set)
genre_tags_ = model._encode_genre_tkns(genre_tokens_)


test_reader = utils.DataReader(vocab=model.vocab)
test_reader.construct(TEST_PATH, sent_size=model.config.sequence_length)

with tf.Session() as sess:
    model.saver.restore(sess, SESS_PATH)

    # Need to construct the feed_dict by hand !!
    for exchange in range(test_reader.nexchange // 2):
        ab_index = 2 * exchange
        encoder_inputs = np.tile(test_reader.encode_inputs[ab_index], (model.config.batch_size, 1)).T
        decoder_targets = np.tile(test_reader.decode_targets[ab_index], (model.config.batch_size, 1)).T
        feed_dict = {model.encoder_inputs: encoder_inputs,
                     model.decoder_targets: decoder_targets,
                     model.genre_tags: genre_tags_}
        log_perp_, weights_ = sess.run([model.generated_batch_log_perp_loss, model.weights], feed_dict)
        perp_ab = np.exp(log_perp_ / weights_.sum(0))

        genre_index = np.argmin(perp_ab)
        ab_report = perp_ab[genre_index]

        bc_index = 2 * exchange + 1
        encoder_inputs = np.tile(test_reader.encode_inputs[bc_index], (model.config.batch_size, 1)).T
        decoder_targets = np.tile(test_reader.decode_targets[bc_index], (model.config.batch_size, 1)).T
        feed_dict = {model.encoder_inputs: encoder_inputs,
                     model.decoder_targets: decoder_targets,
                     model.genre_tags: genre_tags_}
        log_perp_, weights_ = sess.run([model.generated_batch_log_perp_loss, model.weights], feed_dict)
        perp_bc = np.exp(log_perp_ / weights_.sum(0))
        bc_report = perp_bc[genre_index]
        print('{} {}'.format(ab_report, bc_report))



