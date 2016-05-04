"""
Get the context of (Nelson) words from each document and store them.

"""
import gensim
import sys
import logging
import os.path

import wikicorpus
from process import ProcessData

DEFAULT_DICT_SIZE = 200000

if __name__ == "__main__":
    # Read file path for different resources
    wikipath = sys.argv[1]
    outpath = sys.argv[2]
    nelson_norms = sys.argv[3]

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))


    norms_fsg = ProcessData().read_norms(nelson_norms, [])
    norms = set(norms_fsg.keys())
    print("norm list", len(norms_fsg))


    wiki = wikicorpus.WikiCorpus(wikipath, norms, wsize=10) # create word->word_id mapping, takes almost 8h
    wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=DEFAULT_DICT_SIZE)
   #TODO This filtering might remove some of the norms. Experiment with this.

   # save dictionary and bag-of-words (term-document frequency matrix)
    gensim.corpora.MmCorpus.serialize(outpath, wiki, progress_cnt=10000)
    wiki.dictionary.save_as_text(outpath + '_wordids.txt.bz2')
    logger.info("finished running %s" % program)




