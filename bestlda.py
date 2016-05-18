"""
Run the evaluation methods.
"""
#import sys
import numpy as np
import argparse
from collections import defaultdict
from evaluate import Evaluation
from process import ProcessData
import os
import multiprocessing


def defaultdict_list():
    return defaultdict(list)


def eval_ldaworker(arguments):
    args, process, evaluate, filename, norms_fsg, common_words, cuetarget_pairs = arguments
    print("processing %s " % filename)

    norm2doc_path = args.wikiwords.split("_")[0] + ".norm2doc"
    corpus_path =  args.wikiwords.split("_")[0]
    lda_scores = process.read_lda(args.ldapath + filename, norm2doc_path, corpus_path, norms_fsg, common_words)

    asyms = evaluate.asymmetry(lda_scores, pairs)

    te = evaluate.traingle_inequality_threshold(tuples, lda_scores, common_words)

    evaluate.plot_traingle_inequality(te[0], process.outdir + filename + "_")

    assocs = process.get_ct_scores(lda_scores, cuetarget_pairs)

    return asyms, te, assocs, filename

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("nelson", type=str, help="Input Neslon norms path or pickle file.")
    argparser.add_argument("word2vec", type=str, help="Input word2vec path or pickle file.")
    argparser.add_argument("ldapath", type=str, help="the LDA models")
    argparser.add_argument("wikiwords", type=str, help="wiki words file")
    argparser.add_argument("commonwords", type=str, help="common words pickle")
    argparser.add_argument("outdir", default='', help="Directory to place output files. (default='')")
    args = argparser.parse_args()

    process = ProcessData(args.nelson, args.word2vec, args.wikiwords, args.outdir, args.commonwords)
    norms_fsg = process.norms_fsg
    word2vec_cond = process.word2vec_cond
    word2vec_cos = process.word2vec_cos
    common_words = process.common_words

    evaluate = Evaluation()
    pairs = process.get_pairs(norms_fsg, common_words)
    print("number of asym pairs", len(pairs))

    tuples = process.get_tuples(norms_fsg, common_words)
    print("Number of TE tuples", len(tuples))

    evallist = [("norms", norms_fsg), ("word2vec-cond", word2vec_cond), ("word2vec-cos", word2vec_cos)]
    asyms = defaultdict(defaultdict_list)

    for stype, scores in evallist:
        if stype == "word2vec-cos": continue
        asyms["ratio"][stype], asyms["difference"][stype] = evaluate.asymmetry(scores, pairs)
            #if stype == "difference": continue
            #for index in range(len(pairs)):
            #    if asyms[b]["norms"][index] > 30 and \
            #        (asyms[b][stype][index] < 1):
            #            print(pairs[index], asyms[b]["norms"][index], asyms[b][stype][index])

    #Examine whether the traingle inequality holds
    #thresholds = {}
    #thresholds["norms"] = np.arange(0.45, 0.85, 0.1)
    #thresholds["word2vec-cos"] = np.arange(0.75, 1, 0.1)
    #thresholds["word2vec-cond"] =  np.arange(0.0002, 0.00035, 0.00005)#np.arange(0.00035, 0.00051, 0.00005)
    te = defaultdict(defaultdict_list)
    for stype, scores in evallist:
        te_dist, te["ratios"][stype], te["differences"][stype] = evaluate.traingle_inequality_threshold(tuples, \
                scores, common_words)
        evaluate.plot_traingle_inequality(te_dist, process.outdir + stype + "_")

    cuetarget_pairs = pairs#process.get_ct_pairs(norms_fsg, common_words)
    print("Number of cue target pairs % d" % len(cuetarget_pairs))
    assocs = defaultdict_list()
    for stype, scores in evallist:
        assocs[stype] = process.get_ct_scores(scores, cuetarget_pairs)

    # Evaluating different LDA models and finding the best one
    stype=None
    scores=None

    #args, process, evaluate, filename, norms_fsg, common_words = arguments
    pool_input = []
    for filename in os.listdir(args.ldapath):
        if not filename.startswith("topics") or filename.endswith("state") or filename.endswith("pickle"): continue
        pool_input.append((args, process, evaluate, filename, norms_fsg, common_words, cuetarget_pairs))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for reasyms, rete, reassoc, filename  in pool.map(eval_ldaworker, pool_input):
        asyms["ratio"][filename], asyms["difference"][filename] = reasyms
        te_dist, te["ratios"][filename], te["differences"][filename] = rete
        assocs[filename] =  reassoc

    pool.close()
    pool.join()



    print("Asymmetries")
    for b in asyms:
        lda_asym_correlations = []
        for stype in asyms[b]:
            rho = evaluate.rank_correlation(asyms[b]["norms"], asyms[b][stype])
            if stype.startswith("topics"):
                lda_asym_correlations.append((rho[0], rho[1], stype))
            else:
                print("correlation between norms and %s (%s of asymmetries): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))
        lda_asym_correlations.sort(key=lambda tup: tup[0], reverse=True)
        rho = lda_asym_correlations[0]
        print("correlation between norms and lda %s (%s of asymmetries): (%.2f, %.2f)" % (rho[2], b, rho[0], rho[1]))
        print(lda_asym_correlations[:5])
        print()
    lda_asym_correlations = None

    print("Triangle Inequality")
    for b in te:
        lda_te_correlations = []
        for stype in te[b]:
            rho = evaluate.rank_correlation(te[b]["norms"], te[b][stype])
            if stype.startswith("topics"):
                lda_te_correlations.append((rho[0], rho[1], stype))
            else:
                print("correlation between norms and %s (%s of traingle inequality): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))
        lda_te_correlations.sort(key=lambda tup: tup[0], reverse=True)
        rho = lda_te_correlations[0]
        print("correlation between norms and lda %s (%s of traingle inequality): (%.2f, %.2f)" % (rho[2], b, rho[0], rho[1]))
        print(lda_te_correlations[:5])
        print()
        #if stype == "difference": continue
            #for index in range(len(tuples)):
            #    if te[b]["norms"][index] > 30 and \
            #            te[b][stype][index] < 1:
            #                print(tuples[index], te[b]["norms"][index], te[b][stype][index])

    lda_te_correlations = None

    print("Associations")
    print("Overal correlations among associations")
    #
    lda_assoc_correlations = []
    for stype in assocs:
        rho = evaluate.rank_correlation(assocs["norms"], assocs[stype])
        if stype.startswith("topics"):
            lda_assoc_correlations.append((rho[0], rho[1], stype))
        else:
            print("correlation between norms and %s: (%.2f, %.2f)" % (stype, rho[0], rho[1]))
    lda_assoc_correlations.sort(key=lambda tup: tup[0], reverse=True)
    rho = lda_assoc_correlations[0]
    print("correlation between norms and lda %s: (%.2f, %.2f)" % (rho[2], rho[0], rho[1]))
    print(lda_assoc_correlations[:5])
    print()

    # Median rank of associates
    # Sort the norm associates
    print("Median Rank")
    gold_associates = evaluate.sort_scores(norms_fsg)
    for stype, scores in evallist:
        # Sort the word2vec asscociates
        scores_sorted = evaluate.sort_scores(scores)
        print(stype)
        ranks = evaluate.median_rank(gold_associates, scores_sorted, common_words, 5)
        for rank in ranks:
            print("median rank associate %d: %.2f" % (rank+1, np.median(ranks[rank])))
        print
        count = 0
        for cue in scores_sorted:
            print(stype, cue, scores_sorted[cue][:2], gold_associates[cue][:2])
            if count > 4: break
            count += 1









