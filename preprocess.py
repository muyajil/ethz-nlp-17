import nltk
from nltk.tokenize import word_tokenize
import pickle
import sys
import os

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"

def cut_and_pad_sentence(list_of_tokens, length):
    body_len = length-2
    trimmed = list_of_tokens[:body_len]
    padding = [PAD] * (body_len - len(trimmed) )

    munged = [BOS]
    munged.extend(trimmed)
    munged.extend(padding)
    munged.extend([EOS])

    return munged

# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
def flatten(l):
    return [item for sublist in l for item in sublist]

def unk_and_int(sentences, most_common_k):
    all_words = flatten(sentences)
    topk = nltk.FreqDist(all_words).most_common(most_common_k)
    topk_list = [word for (word,count) in topk]
    topk_set = set(topk_list)
    topk2int = dict((x,i) for (i,x) in enumerate(topk_list))
    topk2int[UNK] = len(topk2int)+1

    ret = []
    for sent in sentences:
        new_sent = []
        for word in sent:
            if word not in topk_set:
                word = UNK
            intified = topk2int[word]
            new_sent.append(intified)
        ret.append(new_sent)

    return ret

def read_trim_pad(filename, sentence_length=30):
    sentences = []
    with open(filename) as thefile:
        for line in thefile:
            tokens = word_tokenize(line)
            munged = cut_and_pad_sentence(tokens, sentence_length)
            sentences.append(munged)

    return sentences


def main():
    path = sys.argv[1]
    path = os.path.expanduser(path)

    data = read_trim_pad(path)
    data = unk_and_int(data, 20000)

    outpath = '/tmp/nlu-project-data.pickle'
    print('writing to %s' % outpath)
    with open (outpath, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
