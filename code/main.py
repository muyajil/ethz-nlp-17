'''
Get data and models and train them
'''

import os
import argparse
from data import NlpData
# from models import NlpModel

# Global Constants
DIR = os.path.dirname(os.path.abspath(__file__))
PARSER = argparse.ArgumentParser(description='LSTM Implementation in Tensorflow')

def main(args):
    data_train = NlpData(os.path.join(DIR, args.data_dir), args.file_name, args.vocab_size, args.seq_length)
    batch = data_train.get_next_batch(args.batch_size)
    # TODO: Train model
    # TODO: Serialize Data for Tensorboard

if __name__ == '__main__':
    PARSER.add_argument("--predef", help="Predefined mode for all arguments", action='store_true')
    PARSER.add_argument("--data_dir", help="Directory where you keep your data (relative to where your main.py lives)", type=str)
    PARSER.add_argument("--summaries_dir", help="Directory where the summaries for Tensorboard will be stored", type=str)
    PARSER.add_argument("--file_name", help="The file name of the data to parse", type=str)
    PARSER.add_argument("--batch_size", help="Batch size for the RNN", type=int)
    PARSER.add_argument("--seq_length", help="Sequence Length of the RNN", type=int)
    PARSER.add_argument("--embedding_size", help="Embedding Size of the words", type=int)
    PARSER.add_argument("--hidden_state", help="The size of the hidden states of the RNN", type=int)
    PARSER.add_argument("--vocab_size", help="The size of your vocabulary", type=int)
    args = PARSER.parse_args()
    
    if args.predef:
        if args.data_dir is None:
            args.data_dir = os.path.join(os.path.dirname(DIR), 'data')
        if args.summaries_dir is None:
            args.summaries_dir = os.path.join(os.path.dirname(DIR), 'summaries')
        if args.file_name is None:
            args.file_name = 'sentences.train'
        if args.batch_size is None:
            args.batch_size = 64
        if args.seq_length is None:
            args.seq_length = 30
        if args.embedding_size is None:
            args.embedding_size = 100
        if args.hidden_state is None:
            args.hidden_state = 512
        if args.vocab_size is None:
            args.vocab_size = 20000
    else:
        exists_missing_arg_values = len(list(filter(lambda x: x is None, vars(args).values()))) > 1
        if exists_missing_arg_values and args.predef is None:
            PARSER.error("Please provide all the needed arguments or use --predef")
        else:
            PARSER.error("You did some weird stuff with the arguments, probably forgot one...")
            
    main(args)
