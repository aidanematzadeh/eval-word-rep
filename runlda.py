# Reads the mm corpus, run the online LDA, and grid search for the parameters.

#argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=256, help="Batch size. (default=256)")
#argparser.add_argument("-d", "--num_docs", dest="num_docs", type=int, default=7990787, help="Total # docs in dataset. (default=7990787)")
#argparser.add_argument("-k", "--num_topics", dest="num_topics", type=int, default=100, help="Number of topics. (default=100)")
#argparser.add_argument("-t", "--tau_0", dest="tau_0", type=int, default=1024, help="Tau learning parameter to downweight early documents (default=1024)")
#argparser.add_argument("-l", "--kappa", dest="kappa", type=int, default=0.7, help="Kappa learning parameter; decay factor for influence of batches.(default=0.7)")
#argparser.add_argument("-m", "--model_out_freq", dest="model_out_freq", type=int, default=10000, help="Number of iterations interval for outputting a model file. (default=10000)")
import gensim

import logging
import argparse
import os
import numpy
import multiprocessing
import itertools
import pickle
import sys

def get_chunks(iterable, chunks=1):
    lst = list(iterable)
    print(lst)
    lst = [lst[i::chunks] for i in range(chunks) if len(lst[i::chunks]) > 0]

    print(lst)
    return lst

marked = set([])
def ldaworker(arguments):
    print("arguments", arguments)
    pairs, args = arguments
    corpus = gensim.corpora.MmCorpus(args.corpus)
    id2word = gensim.corpora.Dictionary.load_from_text(args.vocabs)

    print("------- before for loop")
# num_topics=100, id2word=None, distributed=False, chunksize=2000, passes=1, update_every=1 (batch or online),
# alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, minimum_probability=0.01):

    #lda = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=args.num_topics)
    for num_topics, batch_size in pairs:#, tau, kappa in pairs:
        print("--------in the loop")
        tau = 1.0 #offset
        kappa = 0.5 #decay
        if (num_topics, batch_size, tau, kappa) in marked: continue
        #
        print((num_topics, batch_size, tau, kappa))
        print("------- after if statement")
        fname = "topics-%d-bsize-%d-tau-%f-kappa-%f" % (num_topics, batch_size, tau, kappa)
        #
        logger = logging.getLogger('LDA Worker %d' % os.getpid())
        # create file handler which logs even debug messages
        fh = logging.FileHandler(args.outdir + '/ldaworker-%d-%s.log' % (os.getpid(), fname) )
        fh.setLevel(logging.INFO)
         # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info("corpus info ", corpus)
        logger.info("vocab info ", id2word)
        logger.info("topics-%d-bsize-%d-tau-%f-kappa-%f" % (num_topics, batch_size, tau, kappa))
        #
        print("creating the model")
        #
        lda = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, chunksize=batch_size, eval_every=200)
        lda.save(args.outdir + "/" + fname)
        #lda.print_topics(10)
        print("saved the model")
        logger.info("saved the model")
        #
        marked.add((num_topics, batch_size, tau, kappa))
        with open('marked_params.pickle' ,'wb') as output:
            pickle.dump(marked, output)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus", type=str, help="Input corpus filename.")
    argparser.add_argument("vocabs", help="Vocabulary filename.")
    argparser.add_argument("-o", "--outdir", default='', help="Directory to place output files. (default='')")
    args = argparser.parse_args()

    logger = logging.getLogger('LDA Master')
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('ldamaster.log')
    fh.setLevel(logging.INFO)
     # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


    # Parameter search
    num_topics = numpy.arange(10, 100, 100) # 90
    batch_size = numpy.arange(1, 100, 100)
    tau = numpy.arange(1, 100, 10)
    kappa = numpy.arange(0.5, 1, 0.1)

    pairs = itertools.product(num_topics, batch_size)#, tau, kappa)
    logger.info("number of parameters: %d" % len(num_topics) * len(batch_size))
    chunked_pairs = get_chunks(pairs, chunks=(multiprocessing.cpu_count()-2))
    logger.info("chunked_pairs %d" % len(chunked_pairs))
    print(list(zip(chunked_pairs, [args]*len(chunked_pairs))))
    ldaworker( (chunked_pairs[0], args))

    sys.exit(0)

    #pool = multiprocessing.Pool()
    #results = pool.map(ldaworker, zip(chunked_pairs, [args]*len(chunked_pairs)))
    #pool.close()
    #pool.join()

#    print(results)
    # Now combine the results
#    sorted_results = reversed(sorted(results, key=lambda x: x[0]))
#    print next(sorted_results)  # Winner

