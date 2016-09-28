"""
For every word (in the given list), collect context words from the given document.
Also, generate a set of negative examples for each word.
Maps words to ids and keeps the dictionary.
"""

import sys
#import numpy as np
import collections
from collections import defaultdict

def cumulative_dist(vocab, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.
        Adopted from gensim.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.
        """
        vocab_size = len(vocab)
        cum_dist = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        normalization = float(sum([self.vocab[word].count**power for word in self.vocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.vocab[self.index2word[word_index]].count**power / train_words_pow
            self.cum_table[word_index] = round(cumulative * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain


def generate_negative_examples():

def get_positive_context(input_file, wsize):
    global_counts = defaultdict(lambda: defaultdict(int))
    dictionary = dict()
    with open(input_file) as f:
        for counter, doc in enumerate(f):
            if ((counter+1) % 3 == 0):
                print("document processed %d" % counter)
                print("number of vocabulary %d" % len(dictionary))
                #print(sorted(dictionary.keys()))
                break #TODO

            # Getting the words and adding padding
            words = doc.split()
            words = ["_PAD_"] * wsize + words + ["_PAD"] * wsize
            # Collecing the context of each words
            contexts = defaultdict(str)
            for index in range(len(words)):
                w = words[index]
                contexts[w] += " "+ ' '.join(words[index-wsize:index+wsize+1])

            contexts.pop("_PAD_", None)
            # Counting the words in the context of each word and adding them to its global counts
            for w in contexts:
                # Counting the context words
                context_counts = collections.Counter(contexts[w].split())
                # Removing the sentence paddings
                context_counts.pop("_PAD_", None)
                if not w in dictionary:
                    dictionary[w] = len(dictionary)
                w_index = dictionary[w]
                for cw, count in context_counts.items():
                    if not cw in dictionary:
                        dictionary[cw] = len(dictionary)
                    cw_index = dictionary[cw]
                    global_counts[w_index][cw_index] += count
                #    print(w_index, w, cw, global_counts[w_index][cw_index])
    return global_counts, dictionary

if __name__ == "__main__":

    # Read file path for different resources
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    wsize = 5 # The size of the context window from each side

    global_counts, word2index = get_positive_context(input_file, wsize)

    vocab_count = [global_counts[i][i] for i in global_counts]
    print("vocab count", vocab_count)

    words_str= ""
    # Writing the positive examples and the dictionary
    idf = open(output_file + "postive_ids", 'w')
    cntf= open(output_file + "postive_counts", 'w')
    for w in word2index.keys():
        w_index = word2index[w]
        w_ids = ""
        w_counts = ""
        words_str += "%s %d %d\n" % (w, w_index, global_counts[w_index][w_index])
        for cw_index in global_counts[w_index]:
            w_ids += "%d " % cw_index
            w_counts += "%d " % global_counts[w_index][cw_index]
        idf.write(w_ids + "\n")
        cntf.write(w_counts + "\n")
    idf.close()
    cntf.close()

    with open(output_file + "word2id", 'w') as w2idf:
        w2idf.write(words_str)

"""
        print("start writing context in file")
        for word in context_word:
            if len(context_word[word]) > 0:
                context_file = open(context_dir + "/" + word, 'a')
                context_file.write('\n'.join(context_word[word]))
                context_word[word] = []
                context_file.close()

        #break
"""






