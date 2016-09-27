"""
For every word (in the given list), collect context words from the given document.
Also, generate a set of negative examples for each word.
Maps words to ids and keeps the dictionary.
"""

import sys
import numpy as np
import collections
from collections import defaultdict

#check the number of unique words

if __name__ == "__main__":

    # Read file path for different resources
    input_file = sys.argv[1]
#    context_dir = sys.argv[3]

    wsize = 5 # The size of the context window from each side
    global_counts = defaultdict(lambda: defaultdict(int))
    dictionary = dict()
    with open(input_file) as f:
        for counter, doc in enumerate(f):
            if ((counter+1) % 100 == 0):
                print("document processed %d" % counter)
                print("number of vocabulary %d" % len(dictionary))
                #print(sorted(dictionary.keys()))

            nodes = []
            # Adding padding to the beginning and end of the text
            for i in range(0, wsize):
                nodes.append("_PAD_")
            nodes += doc.split()
            for i in range(0, wsize):
                nodes.append("_PAD_")
            # Collecing the context of each words
            contexts = dict()
            for index in range(len(nodes)):
                w = nodes[index]
                if not w in contexts:
                    contexts[w] = ""
                contexts[w] += " "+ ' '.join(nodes[index-wsize:index+wsize+1])

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
                #print(w, context_counts)
                for cw, count in context_counts.items():
                    if not cw in dictionary:
                        dictionary[cw] = len(dictionary)
                    cw_index = dictionary[cw]
                    global_counts[w_index][cw_index] += count
                    #print(w_index, w, cw, global_counts[w_index][cw_index])
            #
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






