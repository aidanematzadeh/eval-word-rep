"""
Run the evaluation methods for a discriminitive topic skipgram model.
"""
import numpy as np
import argparse
#import os
#import multiprocessing

import process
import evaluate

# python3.5 eval_dtsg.py test_results/dec30pos/norms.pickle data/nelson_norms/
# test_results/dec30pos/cbowcos.pickle test_results/dec30pos/cbowcond.pickle
# data/GoogleNews-vectors-negative300.bin test_results/dec30pos/sgcos.pickle
# test_results/dec30pos/sgcond.pickle
# test_results/stephan/size-200_window-5_mc-0_workers-12_sg-0_neg-0_hs-1
# test_results/dec30pos/lda.pickle data/wikipedia_sw_norms_100k/5w_word2id
# test_results/dec30pos/gamma4916 test_results/dec30pos/lambda100154 none
# test_results/dec30pos/allpairs_sg.pickle test_results/dec30pos/


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--norms_pickle", type=str, help="Input Neslon norms pickle file.")
    argparser.add_argument("--norms_dirpath", type=str, default=None, help="Input Neslon norms dir.")
    #
    argparser.add_argument("--cbowcos_pickle", type=str, help="Input cbow cosnie score pickle file.")
    argparser.add_argument("--cbowcond_pickle", type=str, help="Input cbow cond prob pickle file.")
    argparser.add_argument("--cbow_binarypath", type=str, default=None, help="Input cbow Google binary path.")
    #
    argparser.add_argument("--sgcos_pickle", type=str, help="Input skigpram cosnie score pickle file.")
    argparser.add_argument("--sgcond_pickle", type=str, help="Input skipgram cond prob pickle file.")
    argparser.add_argument("--sg_path", type=str, default=None, help="Input skipgram model.")
    #
    argparser.add_argument("--lda_pickle", type=str, help="Input lda cond prob pickle file.")
    argparser.add_argument("--ldavocab_path", type=str, help="Input the LDA word2id filename")
    argparser.add_argument("--ldagamma_path", type=str, help="Input the LDA gamma filename")
    argparser.add_argument("--ldalambda_path", type=str, help="Input the LDA lambda filename")
    argparser.add_argument("--ldamu_path", type=str, default=None, help="Input the LDA mu filename")
    #
    argparser.add_argument("--glovecos_pickle", type=str, help="Input GloVe cosine score pickle file.")
    argparser.add_argument("--glovecond_pickle", type=str, help="Input GloVe cond prob pickle file.")
    argparser.add_argument("--glove_path", type=str, default=None, help="Input GloVe model.")
    #
    argparser.add_argument("--allpairs_pickle", type=str, help="all pairs output name or the pickle file")
    argparser.add_argument("--outdir", default='', help="Directory to place output files. (default='')")
    args = argparser.parse_args()
    if args.ldamu_path == "none": #TODO
        args.ldamu_path = None

    norms = process.get_norms(args.norms_pickle, args.norms_dirpath)
    cbow_cos, cbow_cond = process.get_w2v(args.cbowcos_pickle, args.cbowcond_pickle, norms, args.cbow_binarypath, True)

    sg_cos, sg_cond = process.get_w2v(args.sgcos_pickle,args.sgcond_pickle, norms, args.sg_path, False)

    lda = process.get_lda(args.lda_pickle, norms, args.ldavocab_path,
                          args.ldalambda_path, args.ldagamma_path,
                          args.ldamu_path)

    glove_cos, glove_cond =  process.get_glove(args.glovecos_pickle, args.glovecond_pickle, args.glove_path, norms)


    # Find the common pairs among the different models
    allpairs = process.get_allpairs(args.allpairs_pickle, norms, cbow_cos,
                                    sg_cos, lda, glove_cos)
    asympairs = process.get_asym_pairs(norms, allpairs)
    print("common pairs: %d, asym pairs: %d" % (len(allpairs), len(asympairs)))

    commonwords = set(lda.keys()) & set(cbow_cos.keys()) & \
        set(sg_cos.keys()) & set(glove_cos.keys()) & set(norms.keys())
    print("common cues", len(commonwords))

    tuples = process.get_tuples(norms, allpairs)
    print("Number of TE tuples", len(tuples))

    # List of models to run the evaluation tasks on
    evallist = [("norms", norms),
                ("bin-cbow-cond", cbow_cond),
                ("bin-cbow-cos", cbow_cos),
                ("sg-cond", sg_cond),
                ("sg-cos", sg_cos),
                ("lda", lda),
                ("glove-cos", glove_cos),
                ("glove-cond", glove_cond)]


    print("Asymmetries")
    asyms = {} #defaultdict(lambda: defaultdict(list))
    asyms["ratio"], asyms["difference"] = {}, {}
    for stype, scores in evallist:
        print(stype)
        if stype.endswith("cos"): continue
        asyms["ratio"][stype], asyms["difference"][stype] = evaluate.asymmetry(scores, asympairs)
        #print(sorted(asyms["ratio"][stype], reverse=True)[1:10])
        for index in range(len(asympairs)):
            if asyms["ratio"]["norms"][index] > 30 and (asyms["ratio"][stype][index] < 1):
                print(asympairs[index], asyms["ratio"]["norms"][index], asyms["ratio"][stype][index])

    for b in asyms:
        if b == "difference": continue
        for stype in asyms[b]:
            rho = evaluate.rank_correlation(asyms[b]["norms"], asyms[b][stype])
            print("correlation between norms and %s (%s of asymmetries): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))

    assocs = {} #defaultdict(lambda: defaultdict(list))
    for stype, scores in evallist:
        assocs[stype] = process.get_pair_scores(scores, allpairs)

    print("Associations")
    print("Overal correlations among associations")
    for stype in assocs:
        rho = evaluate.rank_correlation(assocs["norms"], assocs[stype])
        print("correlation between norms and %s: (%.2f, %.2f)" % (stype, rho[0], rho[1]))

    print("Median Rank")
    gold_associates = evaluate.sort_pairs(norms, allpairs)

    for stype, scores in evallist:
        scores_sorted = evaluate.sort_pairs(scores, allpairs)
        print(stype)
        ranks, maxranks = evaluate.median_rank(gold_associates, scores_sorted, 3)
        for rank in ranks:
            print("median rank associate %d: %.2f, median max rank: %.2f" %\
                    (rank+1, np.median(ranks[rank]), np.median(maxranks[rank])))
        print
        count = 0
        for cue in scores_sorted:
            print(stype, cue, scores_sorted[cue][:2], "gold", gold_associates[cue][:2])
            if count > 4: break
            count += 1


    for stype, scores in evallist[1:]:
        scores_sorted = evaluate.sort_all(scores, norms, commonwords)
        print(stype)
        ranks, maxranks = evaluate.median_rank(gold_associates, scores_sorted, 3)
        for rank in ranks:
            print("median rank associate %d: %.2f, median max rank: %.2f" %\
                    (rank+1, np.median(ranks[rank]), np.median(maxranks[rank])))
        print
        count = 0
        for cue in scores_sorted:
            print(stype, cue, scores_sorted[cue][:2], "gold", gold_associates[cue][:2])
            for index,(c,v) in enumerate(scores_sorted[cue]):
                if gold_associates[cue][0][0]==c:
                    print(index, c,v)
            if count > 4: break
            count += 1



    print("Triangle Inequality")
    #Examine whether the traingle inequality holds
    te = {}#defaultdict(lambda: defaultdict(list))

    te["ratio"], te["difference"] = {}, {}
    for stype, scores in evallist:

        te_dist, te["ratio"][stype], te["difference"][stype] = evaluate.traingle_inequality_threshold(tuples, scores, commonwords)
        evaluate.plot_traingle_inequality(te_dist, args.outdir + stype + "_")
    for b in te:
        if b == "difference": continue
        for stype in te[b]:
            rho = evaluate.rank_correlation(te[b]["norms"], te[b][stype])
            print("correlation between norms and %s (%s of traingle inequality): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))

            #for index in range(len(tuples)):
            #    if te[b]["norms"][index] > 30 and \
            #            te[b][stype][index] < 1:
            #                print(tuples[index], te[b]["norms"][index], te[b][stype][index])
