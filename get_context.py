"""
Run the evaluation methods.
"""
import time
import sys
import numpy as np
import os
import cProfile
import re
from io import StringIO
import pstats
import string

from process import ProcessData
import constants as const


if __name__ == "__main__":
    
    # Read file path for different resources
    nelson_norms = sys.argv[1]
    one_billion_dir = sys.argv[2]
    context_dir = sys.argv[3]
    
    translator = str.maketrans({key: None for key in string.punctuation})

    process = ProcessData()

    norms_fsg = process.read_norms(nelson_norms, [])
    norms = set(norms_fsg.keys())
    print("norm list", len(norms_fsg))

    pr = cProfile.Profile()
    
    wsize = 10 # The size of the context window
    hwsize = wsize//2

    context_word = {}
    for word in norms:
        context_word[word] = []

    for filename in os.listdir(one_billion_dir):
        pr.enable()

        print(filename)

        onebillion_file = open(one_billion_dir + "/" + filename, 'r', encoding = "ISO-8859-1")
        for lineno, line in enumerate(onebillion_file):
            nodes = []
            # Adding padding to the beginning of the text
            for i in range(0, hwsize):
                nodes.append("text_start")
            
            # Removing punctuations
            #print(line)
            line = line.translate(translator)
            #print(line)
            # Remove numbers?

            nodes = line.strip().split(' ')

            for i in range(0, wsize):
                nodes.append("text_end")

            for index in range(len(nodes)):
                node = nodes[index]
                if not (node in norms): 
                    continue
            
                #context_word[node].append(' '.join(nodes[index-hwsize:index] + nodes[index+1:index+hwsize+1])) 
                context_word[node].append(' '.join(nodes[index-hwsize:index+hwsize+1])) 
            
            if lineno % 10000 == 0:
                print("got context", lineno)
        
        print("start writing context in file")
        for word in context_word:
            if len(context_word[word]) > 0:
                context_file = open(context_dir + "/" + word, 'a')
                context_file.write('\n'.join(context_word[word]))
                context_word[word] = []
                context_file.close()
        print("writing context was done")    
        
        pr.disable()
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

    
        
        #break







