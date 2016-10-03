"""
Run the evaluation methods for a discriminitive topic skipgram model.
"""
import numpy as np
import argparse
from collections import defaultdict
import os
import multiprocessing

import process
import evaluate

def eval_ldaworker(arguments):
    args, process, evaluate, filename, norms, common_words, cuetarget_pairs = arguments
    norm2doc_path = args.wikiwords.split("_")[0] + ".norm2doc"
    corpus_path =  args.wikiwords.split("_")[0]
    lda_scores = process.read_lda(args.ldapath + filename, norm2doc_path, corpus_path, norms, common_words)
    if lda_scores == None:
        print("lda is none")
        return None, None, None, None

    asyms = evaluate.asymmetry(lda_scores, pairs)

    te = evaluate.traingle_inequality_threshold(tuples, lda_scores, common_words)

    evaluate.plot_traingle_inequality(te[0], process.outdir + filename + "_")

    assocs = process.get_ct_scores(lda_scores, cuetarget_pairs)

    return asyms, te, assocs, filename

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("norms_pickle", type=str, help="Input Neslon norms pickle file.")
    argparser.add_argument("norms_dirpath", type=str, default=None, help="Input Neslon norms dir.")
    argparser.add_argument("cbowcos_pickle", type=str, help="Input cbow cosnie score pickle file.")
    argparser.add_argument("cbowcond_pickle", type=str, help="Input cbow cond prob pickle file.")
    argparser.add_argument("cbow_binarypath", type=str, default=None, help="Input cbow Google binary path.")
    argparser.add_argument("lda_pickle", type=str, help="Input lda cond prob pickle file.")
    argparser.add_argument("ldavocab_path", type=str, help="Input the LDA word2id filename")
    argparser.add_argument("ldagamma_path", type=str, help="Input the LDA gamma filename")
    argparser.add_argument("ldalambda_path", type=str, help="Input the LDA lambda filename")
    #
    argparser.add_argument("allpairs_pickle", type=str, help="all pairs output name or the pickle file")
    argparser.add_argument("outdir", default='', help="Directory to place output files. (default='')")
    args = argparser.parse_args()

    norms = process.get_norms(args.norms_pickle, args.norms_dirpath)
    cbow_cos, cbow_cond = process.get_cbow(args.cbowcos_pickle, args.cbowcond_pickle, norms, args.cbow_binarypath)
    lda = process.get_lda(args.lda_pickle, norms, args.ldavocab_path, args.ldalambda_path, args.ldagamma_path)

    allpairs = process.get_allpairs(args.allpairs_pickle, norms, cbow_cos, lda)
    asympairs = process.get_asym_pairs(norms, allpairs)
    print("all common pairs: %d, asym pairs: %d" % (len(allpairs), len(asympairs)))

    #tuples = process.get_tuples(norms, allpairs)
    #print("Number of TE tuples", len(tuples))


    evallist = [("norms", norms), ("cbow-cond", cbow_cond), ("cbow-cos", cbow_cos), ("lda", lda)]
    #
    asyms = defaultdict(lambda: defaultdict(list))
    for stype, scores in evallist:
        print(stype)
        if stype.endswith("cos"): continue
        asyms["ratio"][stype], asyms["difference"][stype] = evaluate.asymmetry(scores, asympairs)
        print(sorted(asyms["ratio"][stype], reverse=True)[1:10])
        for index in range(len(asympairs)):
            if asyms["ratio"]["norms"][index] > 30 and (asyms["ratio"][stype][index] < 1):
                print(asympairs[index], asyms["ratio"][stype][index], asyms["ratio"][stype][index])
    #
    print("Asymmetries")
    for b in asyms:
        #if b == "difference": continue
        for stype in asyms[b]:
            rho = evaluate.rank_correlation(asyms[b]["norms"], asyms[b][stype])
            print("correlation between norms and %s (%s of asymmetries): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))

    assocs = defaultdict(lambda: defaultdict(list))
    for stype, scores in evallist:
        assocs[stype] = process.get_pair_scores(scores, allpairs)

    print("Associations")
    print("Overal correlations among associations")
    for stype in assocs:
        rho = evaluate.rank_correlation(assocs["norms"], assocs[stype])
        print("correlation between norms and %s: (%.2f, %.2f)" % (stype, rho[0], rho[1]))



    if False:
        #Examine whether the traingle inequality holds
        te = defaultdict(lambda: defaultdict(list))
        for stype, scores in evallist:
            te_dist, te["ratios"][stype], te["differences"][stype] = evaluate.traingle_inequality_threshold(tuples, \
                    scores, common_words)
            evaluate.plot_traingle_inequality(te_dist, process.outdir + stype + "_")
        # Evaluating different LDA models and finding the best one
        stype=None
        scores=None

        #args, process, evaluate, filename, norms, common_words = arguments
        pool_input = []
        for filename in os.listdir(args.ldapath):
            if not filename.startswith("topics") or filename.endswith("state") or filename.endswith("pickle"): continue
            pool_input.append((args, process, evaluate, filename, norms, common_words, cuetarget_pairs))

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for reasyms, rete, reassoc, filename  in pool.map(eval_ldaworker, pool_input):
            if reasyms == None:
                print("reasyms is none")
                continue
            asyms["ratio"][filename], asyms["difference"][filename] = reasyms
            te_dist, te["ratios"][filename], te["differences"][filename] = rete
            assocs[filename] =  reassoc

        pool.close()
        pool.join()

        #import sys
        #sys.exit(0)

        # Select the best parameter setting -- that results in the highest overall association

        # Median rank of associates
        # Sort the norm associates
        print("Median Rank")
        gold_associates = evaluate.sort_scores(norms)
        for stype, scores in evallist:
            # Sort the word2vec asscociates
            scores_sorted = evaluate.sort_scores(scores)
            print(stype)
            ranks = evaluate.median_rank(gold_associates, scores_sorted, common_words, 3)
            for rank in ranks:
                print("median rank associate %d: %.2f" % (rank+1, np.median(ranks[rank])))
            print
            count = 0
            for cue in scores_sorted:
                print(stype, cue, scores_sorted[cue][:2], gold_associates[cue][:2])
                if count > 4: break
                count += 1



        print("Asymmetries")
        for b in asyms:
            if b == "difference": continue
            lda_asym_correlations = []
            for stype in asyms[b]:
                rho = evaluate.rank_correlation(asyms[b]["norms"], asyms[b][stype])
                if stype.startswith("topics"):
                    lda_asym_correlations.append((rho[0], rho[1], stype))
                else:
                    print("correlation between norms and %s (%s of asymmetries): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))
            lda_asym_correlations.sort(key=lambda tup: tup[0], reverse=True)
            rho = lda_asym_correlations
            for index in range(len(rho)):
                if rho[index][2] == best_lda_name:
                    print("correlation between norms and lda %s (%s of asymmetries): (%.2f, %.2f)" % (rho[index][2], b, rho[index][0], rho[index][1]))

            #print(lda_asym_correlations[:5])
            #print()

        lda_asym_correlations = None

        print("Triangle Inequality")
        for b in te:
            if b == "difference": continue
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
            #print(lda_te_correlations[:5])
            #print()
            #if stype == "difference": continue
                #for index in range(len(tuples)):
                #    if te[b]["norms"][index] > 30 and \
                #            te[b][stype][index] < 1:
                #                print(tuples[index], te[b]["norms"][index], te[b][stype][index])

        lda_te_correlations = None









