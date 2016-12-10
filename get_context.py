"""
For every word (in the given list), collect context words from the given document.
Also, generate a set of negative examples for each word.
Maps words to ids and keeps the dictionary.
"""

import sys
import numpy as np
from collections import defaultdict
from stop_words import get_stop_words

def get_stopwords(stopwordsfile):
    stoplist = set(get_stop_words('en'))
    with open(stopwordsfile, 'r') as f:
        stopwords = f.read().split(',')
    for sw in stopwords:
        stoplist.add(sw)
    return stoplist

def cumulative_dist(vocab_counts, power=0.75, domain=2**31 - 1):
    """
    Create a cumulative-distribution table using stored vocabulary word counts for
    drawing random words in the negative-sampling training routines.
    Adopted from gensim.
    """
    vocab_size = len(vocab_counts)
    vocab_counts = np.array(vocab_counts)
    cum_dist = np.zeros(vocab_size, dtype=np.uint32)
    # compute sum of all power (Z in paper)
    normalization = float(sum(vocab_counts**power))
    cumulative = 0.0
    for word_index in range(vocab_size):
        cumulative += vocab_counts[word_index]**power / normalization
        cum_dist[word_index] = round(cumulative * domain)
        #print(word_index, cumulative, round(cumulative * domain))

    if len(cum_dist) > 0:
        assert cum_dist[-1] == domain

    return cum_dist

def get_negative_examples(negative_dist, positive_counts, vocab_size, k=2):
    """
    Draw negative examples given the negative distribution.
    Draw k negative exmples for each postivie one.
    From gensim: to draw a word index, choose a random integer up to the maximum value in the table (negative_dist[-1]).
    then finding that integer's sorted insertion point (as if by bisect_left or ndarray.searchsorted()).
    That insertion point is the drawn index, coming up in proportion equal to the increment at that slot.
    """
    negative_counts = defaultdict(lambda: defaultdict(int))

    for w_index in range(vocab_size):
        # Number of negative examples to generate for w_index.
        # For each word in the context of w_index, k negative examples is generated.
        negative_num = (sum(positive_counts[w_index].values()) - positive_counts[w_index][w_index]) * k
        #print(w_index, "number of negative samples", negative_num)

        positive_examples = set(positive_counts[w_index].keys())

        # Draw negative examples
        negative_examples = []
        rand_vec = np.array(np.random.randint(negative_dist[-1], size=negative_num), dtype='uint32')
        count = 0
        while len(negative_examples) < negative_num:
            nw = negative_dist.searchsorted(rand_vec[count])
            # Consider the word as a negative example if it is not in its context
            if not nw in positive_examples:
                negative_examples.append(nw)

            count += 1
            if count >= negative_num:
                rand_vec = np.array(np.random.randint(negative_dist[-1], size=negative_num), dtype='uint32')
                count = 0

        # Add the negative examples to the negative_counts
        #print(w_index, "negative examples", negative_examples)
        for neg_index in negative_examples:
            negative_counts[w_index][neg_index] += 1

        #assert len(set(negative_examples).intersection(positive_examples)) == 0
        #assert len(negative_examples) == negative_num

    return negative_counts

def get_positive_examples(input_file, wsize, stoplist):
    bigram_counts = defaultdict(lambda: defaultdict(int))
    word_counts = defaultdict(int)

    # mapping between a word and its id
    dictionary = dict()
    reversed_dictionary = dict()

    with open(input_file) as f:
        for counter, doc in enumerate(f):
            if ((counter+1) % 1000 == 0):
                print("document processed %d" % counter)
                print("number of vocabulary %d" % len(dictionary))

            #TODO remove
            if (counter + 1) % 100 == 0:
                break

            words = [token for token in doc.split() \
                    if 2 <= len(token) <= 15 and not token in stoplist]
            doc_len = len(words)
            # Collecing the context of each word in a document
            for index, w in enumerate(words):
                if not w in dictionary:
                    dictionary[w] = len(dictionary)
                    reversed_dictionary[dictionary[w]] = w
                w_index = dictionary[w]
                word_count[w_index] += 1

                context = words[max(0, index-wsize): min(doc_len, index+wsize+1)]
                # Adding context to w's bigram counts
                for cw in context:
                    if not cw in dictionary:
                        dictionary[cw] = len(dictionary)
                        reversed_dictionary[dictionary[cw]] = cw
                    cw_index = dictionary[cw]
                    bigram_counts[w_index][cw_index] += 1
                #    print(w_index, w, cw, bigram_counts[w_index][cw_index])

    return bigram_counts, word_counts, dictionary, reversed_dictionary



if __name__ == "__main__":
    # Read file path for different resources
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    stopword_file = sys.argv[3]
    wsize = 5 # The size of the context window from each side
    vocabulary_size = 100000 #The maximum number of vocabular to keep

    # Read the stop word list
    stoplist = get_stopwords(stopword_file)

    bigram_counts, word_count, word2index, index2word = get_positive_examples(input_file, wsize, stoplist)

    sort_vocab = sorted(word_count.items(), key=lambda x: x[1])

    print("sort_vocab", sort_vocab)

    #cum_dist = cumulative_dist(vocab_counts)
    #negative_counts = get_negative_examples(cum_dist, bigram_counts, len(vocab_counts))

    #for w_index in bigram_counts.keys():
    #    print("neg", w_index, sorted(negative_counts[w_index]))
    #    print("pos", w_index, sorted(bigram_counts[w_index]))


    # Writing the positive examples and the dictionary
    idf = open(output_file + str(wsize) + "w_postive_ids", 'w')
    cntf= open(output_file + str(wsize) + "w_postive_counts", 'w')
    #neg_idf = open(output_file + str(wsize) + "w_negative_ids", 'w')
    #neg_cntf = open(output_file + str(wsize) + "w_negative_counts", 'w')
    words_str= ""
    for w_index in range(len(word2index)):
        w = index2word[w_index]
        words_str += "%s %d %d\n" % (w, w_index, bigram_counts[w_index][w_index])
        # Writing the positive files
        ids, counts = "",""
        for cw_index in bigram_counts[w_index]:
            ids += "%d " % cw_index
            counts += "%d " % bigram_counts[w_index][cw_index]
        idf.write(ids + "\n")
        cntf.write(counts + "\n")
        # Writing the negative files
        ids, counts = "", ""
        #for nw_index in negative_counts[w_index]:
        #    ids += "%d " % nw_index
        #    counts += "%d " % negative_counts[w_index][nw_index]
        #neg_idf.write(ids + "\n")
        #neg_cntf.write(counts + "\n")

    idf.close()
    cntf.close()
    #neg_idf.close()
    #neg_cntf.close()

    with open(output_file + "word2id", 'w') as w2idf:
        w2idf.write(words_str)






