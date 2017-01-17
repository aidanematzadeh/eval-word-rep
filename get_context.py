"""
For every word (in the given list), collect context words from the given document.
Also, generate a set of negative examples for each word.
Maps words to ids and keeps the dictionary.
"""

import sys
import numpy as np
from stop_words import get_stop_words
import process

def get_stopwords(stopwordsfile):
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    stoplist = set(get_stop_words('en') + numbers)
    with open(stopwordsfile, 'r') as f:
        stopwords = f.read().split(',')
    for sw in stopwords:
        stoplist.add(sw)
    return stoplist

def cumulative_dist(word_counts, power=0.75, domain=2**31 - 1):
    """
    Create a cumulative-distribution table using stored vocabulary word counts for
    drawing random words in the negative-sampling training routines.
    Adopted from gensim.
    """
    vocab_size = len(word_counts)
    vocab_counts = np.array([word_counts[index] for index in range(vocab_size)])
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

def get_negative_examples(negative_dist, positive_counts, id2word, k=2):
    """
    Draw negative examples given the negative distribution.
    Draw k negative exmples for each postivie one.
    From gensim: to draw a word index, choose a random integer up to the maximum value in the table (negative_dist[-1]).
    then finding that integer's sorted insertion point (as if by bisect_left or ndarray.searchsorted()).
    That insertion point is the drawn index, coming up in proportion equal to the increment at that slot.
    """
    negative_counts =  dict()

    for wid in positive_counts:
        # Number of negative examples to generate for w_index.
        # For each word in the context of w_index, k negative examples is generated.
        negative_counts[wid] = {}
        #
        #TODO it is possible that the norm is not in the selected 100k words
        negative_num = 0
        for cid in positive_counts[wid]:
            if cid != wid:
                negative_num += positive_counts[wid][cid]
        negative_num *= k
        negative_num = min(500000, negative_num) #TODO

        print("cue: %s freq: %d pos exmples: %d neg examples: %d" % (id2word[wid], positive_counts[wid][wid], len(positive_counts[wid]), negative_num))

        # Draw negative examples
        negative_examples = []
        positive_examples = set(positive_counts[wid].keys())
        rand_vec = np.array(np.random.randint(negative_dist[-1], size=negative_num), dtype='uint32')
        count = 0
        while len(negative_examples) < negative_num:
            negw = negative_dist.searchsorted(rand_vec[count])
            # Consider the word as a negative example if it is not in its context
            if negw in positive_examples:
                negative_examples.append(negw)

            count += 1
            if count >= negative_num:
                rand_vec = np.array(np.random.randint(negative_dist[-1], size=negative_num), dtype='uint32')
                count = 0

        # Add the negative examples to the negative_counts
        for neg_index in negative_examples:
            if not neg_index in negative_counts[wid]:
                negative_counts[wid][neg_index] = 0
            negative_counts[wid][neg_index] += 1

        #assert len(set(negative_examples).intersection(positive_examples)) == 0
        #assert len(negative_examples) == negative_num

    return negative_counts

def get_positive_examples(input_file, wsize, stoplist, norms=None):
    bigram_counts = dict()
    word_counts = dict()

    # Mapping between a word and its id
    dictionary = dict()
    reversed_dictionary = dict()

    with open(input_file) as f:
        for counter, doc in enumerate(f):
            if ((counter+1) % 10000 == 0):
                print("document processed %d" % counter)
                print("number of words  %d" % len(dictionary))

            #TODO remove
            #if (counter + 1) % 2000 == 0:
            #    break

            words = [token for token in doc.split() \
                    if 2 < len(token) <= 15 and not token in stoplist]
            doc_len = len(words)
            # Collecing the context of each word in a document
            for index, w in enumerate(words):
                if not w in dictionary:
                    dictionary[w] = len(dictionary)
                    reversed_dictionary[dictionary[w]] = w
                    word_counts[dictionary[w]] = 0
                w_index = dictionary[w]
                word_counts[w_index] += 1
                #
                # TODO Only keeping context for norms
                if norms != None and (not w in norms): continue
                #
                context = words[max(0, index-wsize): min(doc_len, index+wsize+1)]
                # Adding context to w's bigram counts
                for cw in context:
                    if not cw in dictionary:
                        dictionary[cw] = len(dictionary)
                        reversed_dictionary[dictionary[cw]] = cw
                        word_counts[dictionary[cw]] = 0
                    #
                    cw_index = dictionary[cw]
                    #
                    if not w_index in bigram_counts:
                        bigram_counts[w_index] = {}
                    if not cw_index in bigram_counts[w_index]:
                        bigram_counts[w_index][cw_index] = 0

                    bigram_counts[w_index][cw_index] += 1

    return bigram_counts, word_counts, dictionary, reversed_dictionary



if __name__ == "__main__":
    # Read file path for different resources
    input_file = sys.argv[1]
    stopword_file = sys.argv[2]
    output_file = sys.argv[3]
    norms_path = sys.argv[4]

    norms = process.get_norms(output_file+"/norms.pickle", norms_path)
    print("Number of cues in norms", len(norms.keys()))

    wsize = 5 # The size of the context window from each side
    vocabulary_size = 100000 #The maximum number of vocabular to keep
    stop_freq_limit = 100

    # Read the stop word list
    stoplist = get_stopwords(stopword_file)

    bigram_counts, word_counts, word2index, index2word = get_positive_examples(input_file, wsize, stoplist, set(norms.keys()))

    # Removing the very frequent words as stop words
    sort_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    sort_vocab = sort_vocab[stop_freq_limit: vocabulary_size + stop_freq_limit]

    printlist = [(index2word[wid], freq) for wid, freq in sort_vocab[:100]]
    print("Most frequent words", len(printlist), printlist)

    cuefreq = [(word2index[cue], word_counts[word2index[cue]]) for cue in norms.keys() if cue in word2index.keys()]
    sort_vocab = cuefreq + sort_vocab

     # Reassign ids to words
    new_word_counts, new_bigram_counts = dict(), dict()
    new_word2id, new_id2word = dict(), dict()
    new_id = 0
    for (old_id, freq) in sort_vocab:
        word = index2word[old_id]
        if word in new_word2id.keys(): continue
        new_word2id[word] = new_id
        new_id2word[new_id] = word
        new_word_counts[new_id] = freq
        new_id += 1

    selected_words = set(new_word2id.keys())
    # Only keep positive examples that are selected in the new vocabulary
    for (old_id, wfreq) in cuefreq:
        new_id = new_word2id[index2word[old_id]]
        new_bigram_counts[new_id] = {}
        for old_cid, cfreq in bigram_counts[old_id].items():
            if index2word[old_cid] in selected_words:
                new_bigram_counts[new_id][new_word2id[index2word[old_cid]]] = cfreq

    word_counts, bigram_counts = None, None
    print("Size of vocab", len(new_word2id))
    cum_dist = cumulative_dist(new_word_counts)
    print("Start getting negative examples.")
    negative_counts = get_negative_examples(cum_dist, new_bigram_counts,
                                            new_id2word)


    # Writing the positive examples and the dictionary
    prefix  = output_file + str(wsize) + "w_"
    idf = open(prefix + "positive_ids", 'w')
    cntf= open(prefix + "positive_counts", 'w')
    neg_idf = open(prefix + "negative_ids", 'w')
    neg_cntf = open(prefix + "negative_counts", 'w')
    w2idf =  open(prefix   + "word2id", 'w')
    print("start writing")

    for w, w_index in sorted(new_word2id.items(), key=lambda x:x[1]):
        w2idf.write("%s %d %d\n" % (w, w_index, new_word_counts[w_index]))
    w2idf.close()

    for w_index in sorted(new_bigram_counts.keys()):
        # Writing the positive files
        ids, counts = "%d: " % w_index, "%d: " % w_index
        for cw_index, freq in new_bigram_counts[w_index].items():
            ids += "%d " % cw_index
            counts += "%d " % freq
        idf.write(ids + "\n")
        cntf.write(counts + "\n")
        # Writing the negative files
        ids, counts = "%d: " % w_index, "%d: " % w_index
        #ids, counts = "", ""
        for nw_index, freq in negative_counts[w_index].items():
            ids += "%d " % nw_index
            counts += "%d " % freq
        neg_idf.write(ids + "\n")
        neg_cntf.write(counts + "\n")

    idf.close()
    cntf.close()
    neg_idf.close()
    neg_cntf.close()





