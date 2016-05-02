"""
Get the context of (Nelson) words from each document and store them.
"""
import wikicorpus
import sys
from gensim import MmCorpus

if __name__ == "__main__":
    # Read file path for different resources
    wikipath = sys.argv[1]
    outpath = sys.argv[2]
    wiki = wikicorpus.WikiCorpus(wikipath) # create word->word_id mapping, takes almost 8h
    MmCorpus.serialize(outpath, wiki)







