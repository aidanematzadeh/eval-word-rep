import os
import sys
import multiprocessing
import itertools
import logging
import argparse
import numpy as n

import dtsg
import hofonlineldavb as poslda
import process

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# time python3.5 run_dtsg.py data/wikipedia_sw_norms_100k/5w_
# test_results/jan16pos/ positive online all data/nelson_norms/ | tee
# test_results/jan16pos/log

# stream class is taken from electricmonk.nl
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
     """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())




lock_dir = "/opt/local/amint/anc/lockfiles/"
def ldaworker(arguments):
    pairs, args = arguments
    negative_flag = (args.negflag.lower() == 'neg')
    pos_wordids, pos_wordcts = process.read_tsgdata(args.datapath +
                                                    "positive_counts",
                                                    args.datapath +
                                                    "positive_ids")
    pos_wordids =n.array(pos_wordids)
    pos_wordcts =n.array(pos_wordcts)

    vocab2id, id2vocab = process.read_tsgvocab(args.datapath + "word2id")
    if negative_flag:
        neg_wordids, neg_wordcts = process.read_tsgdata(args.datapath +
                                                        "negative_counts",
                                                        args.datapath +
                                                        "negative_ids")
    # assert(len(pos_wordids) == len(pos_wordcts))
    # assert(len(pos_wordids[0]) == len(pos_wordcts[0]))

    for topic_num, batch_size, tau, kappa, eta, alpha in pairs:
        fname = "topics-%d-bsize-%d-tau-%f-kappa-%f-eta-%f-alpha-%f" %\
             (topic_num, batch_size, tau, kappa, eta, alpha)
        # Number of documents
        doc_num = len(pos_wordids)

        # make sure the run does not exist
        lockfile = (lock_dir + args.negflag + "-" + fname)
        if os.path.exists(lockfile): continue
        open(lockfile,'w').close()

        # create the logger
        logger = logging.getLogger('LDA Worker %d' % os.getpid())
        fh = logging.FileHandler(args.outpath + '/ldaworker-%d-%s.log' %
                                 (os.getpid(), fname))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        rootlog = logging.getLogger()
        rootlog.addHandler(fh)
        rootlog.setLevel(logging.INFO)

        logger.info("number of documents %d" % doc_num)
        logger.info("vocab size %d" %  len(vocab2id))
        logger.info("param info %s" % fname)

        sl = StreamToLogger(logger, logging.INFO)
        sys.stdout = sl

        fname = args.outpath + fname
        epochs = 3

        # creating the model
        if negative_flag:
            logger.info("creating the negative model")
            model = dtsg.OnlineLDA(vocab2id, K=topic_num, D=doc_num, alpha=alpha,
                            eta=eta, zeta=1, tau0=tau, kappa=kappa)
            bounds = model.online_train(pos_wordids, pos_wordcts, neg_wordids,
                               neg_wordcts, fname, batch_size, epochs)
        else:
            logger.info("creating the positive model")
            model = poslda.OnlineLDA(vocab2id, K=topic_num, D=doc_num,
                                     alpha=alpha, eta=eta, tau0=tau,
                                     kappa=kappa)
            bounds = model.online_train(pos_wordids, pos_wordcts, fname,
                                        batch_size, epochs)

        # Plotting bound and perplexity
        plt.plot(bounds)
        plt.savefig(fname +"bound_plot%d.png" % len(pos_wordids))

        # printing topics
        model.print_topics(id2vocab)

        fh.flush()
        ch.flush()


def get_chunks(iterable, chunks=1):
    lst = list(iterable)
    return [lst[i::chunks] for i in range(chunks) if len(lst[i::chunks]) > 0]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("datapath", type=str, help="Input path to positive/negative counts/ids and word2id map")
    argparser.add_argument("outpath", type=str, help="Directory to place output files")
    # argparser.add_argument("normspath", type=str, help="Directory to place output files")

    argparser.add_argument("negflag", help="Use negative (neg) examples or not")
    # argparser.add_argument("batchflag", help="Batch (batch) or online processing")
    # argparser.add_argument("normsflag", help="Use only norms or all vocab")
    args = argparser.parse_args()

    logger = logging.getLogger('LDA Master')
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(args.outpath + '/ldamaster.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s -\
                                  %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


    # Parameter search
    topic_num = [100, 200, 300, 500]  # numpy.arange(20, 100, 20)
    batch_size = [512]  # [1, 4, 16, 64, 256, 512]

    tau = [1]  # [1, 4, 16, 64, 256, 512]
    kappa = [0.5]  # numpy.arange(0.5, 1, 0.1)

    eta = [0.001, 0.0001, 0.000001]
    alpha = [0.01, 0.001, 0.000001]

    pairs = itertools.product(topic_num, batch_size, tau, kappa, eta, alpha)
    logger.info("number of parameters: %d" % (len(topic_num) * len(batch_size) *
                                              len(tau) * len(kappa) *
                                              len(eta) * len(alpha)))

    chunked_pairs = get_chunks(pairs, chunks=(multiprocessing.cpu_count()-1))
    logger.info("chunked pairs %d" % len(chunked_pairs))

    pool = multiprocessing.Pool()
    results = pool.map(ldaworker, zip(chunked_pairs, [args]*len(chunked_pairs)))
    pool.close()
    pool.join()

#export  OMP_NUM_THREADS=1;



#if batch_flag:
#            kappa = 0

# batch_flag = (args.batchflag.lower() == 'batch')

#        if batch_flag:

#            relative_change = 1
#            for counter in range(0, 4):
#                gamma, bound = model.update_lambda(pos_wordids, pos_wordcts)
#                counts_sum = sum(map(sum, pos_wordcts))
#                wbound = bound * len(pos_wordids) / (doc_num * counts_sum)

#                if len(bounds) > 1:
#                    relative_change = bound - bounds[-1]

 #               print("counter", counter)
 #               print("relative_change", relative_change)
 #               print('rho_t = %f,  held-out perplexity estimate = %f, \
 #                   approx bound = %.5f' % (model._rhot, n.exp(-wbound), bound))
 #               perplexity.append(wbound)
 #               bounds.append(bound)
 #       else:

