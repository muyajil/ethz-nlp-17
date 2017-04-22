import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
        help='directory containing the data files')
    parser.add_argument('--save_dir', type=str, defualt='./save',
	help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=512,
	help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
	help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=64,
	help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=30,
	help='RNN sequence length')
    parser.add_argument('num_epochs', type=int, default=50,
	help='number of epochs')
    args = parser.parse_args()

if __name__ == '__main__':
    main()
