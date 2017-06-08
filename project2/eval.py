import os
import sys
import tensorflow as tf
import utils
import numpy as np

from configs import bos_genres, baseline, concat_embedding, concat_one_hot_word_embedding
_bdir = os.path.dirname(os.path.abspath(__file__))

PROMPT_PATH = os.path.join(_bdir, 'data', 'Testing_Prompts.txt')
SESS_PATH = 'project2/models/genre_bos/chatbot.ckpt'
MODEL_TYPE = 'bos_genres'
assert MODEL_TYPE in {'bos_genres', 'baseline', 'concat_embedding', 'concat_one_hot_word_embedding'}


def display(vocab, encoded_input_tokens, gen_preds_, genre_tokens_):
    inputs_str = ' '.join([vocab.decode(x) for x in encoded_input_tokens])
    print('INPUT           > {}'.format(inputs_str))
    for genre_tkn,  preds in zip(genre_tokens_, gen_preds_):
        print('  {:13} > {}'.format(genre_tkn, ' '.join([vocab.decode(x) for x in preds])))
    print()
    return

if MODEL_TYPE == 'baseline':
    config = baseline.config
    Model = baseline.Model
elif MODEL_TYPE == 'bos_genres':
    config = bos_genres.config
    Model = bos_genres.GenreBosSeq2Seq
elif MODEL_TYPE == 'concat_embedding':
    config = concat_embedding.config
    Model = concat_embedding.GenreConcatEmbeddingSeq2Seq
elif MODEL_TYPE == 'concat_one_hot_word_embedding':
    config = concat_embedding.config
    Model = concat_one_hot_word_embedding.GenreConcatOneHotSeq2Seq
else:
    raise ValueError("shouldn't ever get here ... ")


# kind of shitty that we need to load the meta reader just to get num_genres ...
config.batch_size = utils.MetaReader(config.meta_path, config.label_path).num_genres
model = Model(config)

genre_tokens_ = sorted(model.meta_reader.genre_set)
genre_tags_ = model._encode_genre_tkns(genre_tokens_)

with tf.Session() as sess:
    model.saver.restore(sess, SESS_PATH)

    # hacks to get the encoder inputs in right format
    inputs_shape = (model.config.batch_size, model.config.encoder_sequence_length)
    for line in open(PROMPT_PATH, 'r'):
        encoder_inputs_ = np.full(inputs_shape, model.data_reader._pad_token, dtype=int)
        encoded_tokens = np.array([model.vocab.encode(tkn) for tkn in line.strip().split()])
        encoded_line = model.data_reader.preproc_encode_input(encoded_tokens,
            model.config.encoder_sequence_length)
        encoder_inputs_[:, -encoded_line.size:] = encoded_line

        feed_dict = {model.encoder_inputs: encoder_inputs_.T,
                     #model.decoder_inputs: encoder_inputs_.T,  # We won't use decoder inputs,
                     #model.decoder_targets: encoder_inputs_.T, # but the graph won't compile ...
                     model.genre_tags: genre_tags_}

        gen_preds_ = sess.run(model.generated_preds, feed_dict)
        display(model.vocab, encoded_tokens, gen_preds_.T, genre_tokens_)
