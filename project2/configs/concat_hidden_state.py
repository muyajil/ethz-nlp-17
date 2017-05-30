# Augment the final hidden state from the encoder with embeddings of genre information

#TODO: config.encoder_hidden_units < config.decoder_hidden_units
#TODO: create embedding look up table of dim config.decoder_hidden_units - config.encoder_hidden_units
#TODO: overwrite Model.augment_hidden_state to look up embeddings, concat to hidden state


import os
import sys
import tensorflow as tf

# safe (?) way to do sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
