'''
Get data and models and train them
'''

import os
import argparse
from data_handling import dataprovider
from model_handling import modelprovider

# Global Constants
DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(DIR, 'data')
FILE_BASE = 'sentences.'
PARSER = argparse.ArgumentParser(description='LSTM Implementation in Tensorflow')

def main():
    data_train, vocab, targets = dataprovider.get_data(DATA_DIR, FILE_BASE, 'train')
    # TODO: sess.run

if __name__ == '__main__':
    main()
