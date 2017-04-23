import argparse
import tensorflow as tf

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
    parser.add_argument('--grad_clip', type=float, default=10.,
        help='clip gradients at this value')
    parser.add_argument('--voc_size', type=int, default=20000,
	help='size of the vocabulary')
    args = parser.parse_args()

    model(args)

def model():

    # placeholder for the inputs and targets
    words = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
    targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

    # Softmax weights
    softmax_w = tf.Variable(tf.zeros([args.rnn_size, args.voc_size]))
    softmax_b = tf.Variable(tf.zeros([args.voc_size]))

    # embedding = tf.Variable(tf.zeros([args.vocab_size, args.rnn_size])
    # inputs = tf.nn.embedding_lookup(embedding, #TODO: input_data)

    # Cell implementation based on https://arxiv.org/abs/1409.2329
    # Includes a regularization method for RNNs
    lstm = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size)

    # Initial state of the LSTM memory.
    initial_state = state = tf.zeros([args.batch_size, args.rnn_size])
    probabilities = []
    loss = 0.0

    # "loop that sets up the unrolled graph"
    for i in range(args.seq_length):
        output, state = lstm(words[:, i], state)
        # TODO: loss
    final_state = state

'''
    for batch in words_in_dataset:

        # The value of state is updated after processing each batch of words.
        output, state = lstm(batch, state)

        # The LSTM output can be used to make next word predictions
        logits = tf.matmul(output, softmax_w) + softmax_b
        probabilities.append(tf.nn.softmax(logits))
        loss += loss_function(probabilities, target_words)
'''

if __name__ == '__main__':
    main()
