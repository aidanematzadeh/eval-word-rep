"""
For every word (in the given list), collect context words from the given document.
Also, generate a set of negative examples for each word.
Maps words to ids and keeps the dictionary.

Uses filtered TASA corpus.

"""

import sys
import numpy as np
import pickle
import process
import scipy.io
from stop_words import get_stop_words

def get_stopwords(stopwordsfile=None):
    numbers = ["zero", "one", "two", "three", "four", "five",
               "six", "seven", "eight", "nine", "ten"]
    stoplist = set(get_stop_words('en') + numbers)
    # print(stopwordsfile)
    #
    with open(stopwordsfile, 'r') as f:
        stopwords = f.read().split(',')
    for sw in stopwords:
        stoplist.add(sw)

    with open("data/common_words.txt", 'r') as f:
        for line in f:
            stoplist.add(line.strip())

    return stoplist

def get_positive_examples(input_file, id2word, wsize, stopwordfile=None, norms=None):
    bigram_counts = dict()
    word_counts = dict()
    stoplist = get_stopwords(stopwordfile)
    print("size of stop list %d" % len(stoplist))

    document = []

    doc_counts = 0
    with open(input_file) as f:
        for counter, wordid in enumerate(f):
            wordid = int(wordid.strip())

            if wordid == -2:
                continue

            if wordid != -1:
                word = id2word[wordid]

                if len(word) <= 2 or len(word) > 15:
                    continue

                if word in stoplist:
                    continue

                document.append(wordid)
                if wordid not in word_counts.keys():
                    word_counts[wordid] = 0
                word_counts[wordid] += 1

            if wordid == -1 and len(document) > 0:
                # Collecing the context of each word in a document
                doc_len = len(document)
                for index, wid in enumerate(document):
                    w = id2word[wid]
                    if norms != None and (w not in norms):
                        continue
                    context = document[max(0, index-wsize): min(doc_len, index+wsize+1)]
                    if wid not in bigram_counts:
                        bigram_counts[wid] = {}
                    # Adding context to w's bigram counts
                    for cwid in context:
                        if cwid not in bigram_counts[wid]:
                            bigram_counts[wid][cwid] = 0
                        bigram_counts[wid][cwid] += 1

                doc_counts += 1
                document = []

    print("doc count", doc_counts)
    return bigram_counts, word_counts

if __name__ == "__main__":
    # Read file path for different resources
    input_file = sys.argv[1]
    words_file = sys.argv[2]
    output_file = sys.argv[3]
    norms_path = sys.argv[4]
    stopword_file = sys.argv[5]
    print(stopword_file)

    norms = process.get_norms(output_file+"/norms.pickle", norms_path)
    print("Number of cues in norms", len(norms.keys()))

    index2word, word2index = {}, {}
    wordslist = scipy.io.loadmat(words_file)['words']
    for index, word in enumerate(wordslist):
        index2word[index] = word[0][0].lower()
        word2index[word[0][0].lower()] = index

    wsize = 5 # The size of the context window from each side
    #vocabulary_size = 100000 #The maximum number of vocabular to keep
    #stop_freq_limit = 100

    bigram_counts, word_counts = get_positive_examples(input_file, index2word,
                                                       wsize, stopword_file,
                                                       set(norms.keys()))

    sort_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    l = [(index2word[wid], freq) for wid, freq in sort_vocab[:10]]
    print(l)
    l = [(index2word[wid], freq) for wid, freq in sort_vocab[-10:]]
    print(l)

    tokenfreq = sum([freq for index,freq in sort_vocab])
    print("tokenfreq %d typefreq %d" % (tokenfreq, len(sort_vocab)))


    cuefreq = [(word2index[cue], word_counts[word2index[cue]])
               for cue in norms.keys() if cue in word2index.keys()
               if word2index[cue] in word_counts.keys()]

    print("cues in common %d" % len(cuefreq))
    sort_vocab = cuefreq + sort_vocab

    # Reassign ids to words such that norms have the smallest ids
    new_word_counts, new_bigram_counts = dict(), dict()
    new_word2id, new_id2word = dict(), dict()
    new_id = 0
    for (old_id, freq) in sort_vocab:
        word = index2word[old_id]
        # Already added
        if word in new_word2id.keys():
            continue

        new_word2id[word] = new_id
        new_id2word[new_id] = word

        new_word_counts[new_id] = freq
        new_id += 1

    tsgtokenfreq =0
    selected_words = set(new_word2id.keys())
    # Only keep positive examples that are selected in the new vocabulary
    for (old_id, wfreq) in cuefreq:
        new_id = new_word2id[index2word[old_id]]
        new_bigram_counts[new_id] = {}
        for old_cid, cfreq in bigram_counts[old_id].items():
            if index2word[old_cid] in selected_words:
                new_bigram_counts[new_id][new_word2id[index2word[old_cid]]] = cfreq
                tsgtokenfreq += cfreq

    print("tsg token freq", tsgtokenfreq)


    # Writing the positive examples and the dictionary
    prefix  = output_file + str(wsize) + "w_"
    idf = open(prefix + "positive_ids", 'w')
    cntf= open(prefix + "positive_counts", 'w')
    wdf = open(prefix + "wdf", 'w')

    # write the dictionary as txt
    with open(prefix  + "word2id", 'w') as w2idf:
        for w, w_index in sorted(new_word2id.items(), key=lambda x:x[1]):
            w2idf.write("%s %d %d\n" % (w, w_index, new_word_counts[w_index]))
    # write the dictionary as pkl
    with open(prefix + "id2word.pickle", "wb") as f:
        pickle.dump(new_id2word, f)

    doclist = []
    wordlist = []
    for w_index in sorted(new_bigram_counts.keys()):
        # Writing the positive files
        ids, counts = "%d: " % w_index, "%d: " % w_index
        for cw_index, freq in new_bigram_counts[w_index].items():

            # Not including the cue
            # if w_index == cw_index:
            #    continue

            ids += "%d " % cw_index
            counts += "%d " % freq
            wdf.write("%d\t%d\t%d\n" % (cw_index, w_index, freq))

            for item in range(freq):
                doclist.append(w_index)
                wordlist.append(cw_index)

        idf.write(ids + "\n")
        cntf.write(counts + "\n")

    idf.close()
    cntf.close()

    with open(prefix + "D.pickle", "wb") as d:
        pickle.dump(doclist, d)

    with open(prefix + "V.pickle", "wb") as v:
        pickle.dump(wordlist, v)

