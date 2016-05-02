"""
Get the context of (Nelson) words from each document and store them.

"""
import gensim
import wikicorpus
import sys
from process import ProcessData

if __name__ == "__main__":
    # Read file path for different resources
    wikipath = sys.argv[1]
    outpath = sys.argv[2]
    nelson_norms = sys.argv[3]

    norms_fsg = ProcessData().read_norms(nelson_norms, [])
    norms = set(norms_fsg.keys())
    print("norm list", len(norms_fsg))


    wiki = wikicorpus.WikiCorpus(wikipath, norms, wsize=10) # create word->word_id mapping, takes almost 8h
    gensim.corpora.MmCorpus.serialize(outpath, wiki)







