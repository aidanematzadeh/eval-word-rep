"""
Run the evaluation methods for different models.
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
    argparser.add_argument("--cbowcond_eq", type=str, default=None, help="Which equation should be used to compute the conditional probability?")
    #
    argparser.add_argument("--sgcos_pickle", type=str, help="Input skigpram cosnie score pickle file.")
    argparser.add_argument("--sgcond_pickle", type=str, help="Input skipgram cond prob pickle file.")
    argparser.add_argument("--sg_path", type=str, default=None, help="Input skipgram model.")
    argparser.add_argument("--sgcond_eq", type=str, default=None, help="Which equation should be used to compute the conditional probability?")
    #
    argparser.add_argument("--tsg_vocabpath", type=str, help="Input TSG word2id")
    argparser.add_argument("--tsg_countspath", type=str, help="Input TSG POS counts")
    argparser.add_argument("--tsg_idspath", type=str, help="Input TSG POS ids")
    argparser.add_argument("--tsgfreq_pickle", type=str, help="Input TSG freq pickle.")
    #
    argparser.add_argument("--tsgpos_pickle", type=str, help="Input TSGPOS cond prob pickle.")
    argparser.add_argument("--tsgpos_gammapath", type=str, help="Input TSGPOS gamma.")
    argparser.add_argument("--tsgpos_lambdapath", type=str, help="Input TSGPOS lambda.")
    #
    argparser.add_argument("--tsgneg_pickle", type=str, help="Input TSGNEG cond prob pickle.")
    argparser.add_argument("--tsgnegnorm_pickle", type=str, help="Input TSGNEG cond prob pickle.")
    argparser.add_argument("--tsgneg_gammapath", type=str, help="Input TSGNEG gamma.")
    argparser.add_argument("--tsgneg_lambdapath", type=str, help="Input TSGNEG lambda.")
    argparser.add_argument("--tsgneg_mupath", type=str, help="Input TSGNEG mu.")
    #
    argparser.add_argument("--glovecos_pickle", type=str, help="Input GloVe cosine score pickle file.")
    argparser.add_argument("--glovecond_pickle", type=str, help="Input GloVe cond prob pickle file.")
    argparser.add_argument("--glove_path", type=str, default=None, help="Input GloVe model.")
    #
    argparser.add_argument("--allpairs_pickle", type=str, help="all pairs output name or the pickle file")
    argparser.add_argument("--outdir", default='', help="Directory to place output files. (default='')")
    args = argparser.parse_args()

    # Getting different score files
    norms = process.get_norms(args.norms_pickle, args.norms_dirpath)

    cbow_cos, cbow_cond = process.get_w2v(args.cbowcos_pickle,
                                          args.cbowcond_pickle,
                                          norms, args.cbow_binarypath, True,
                                          args.cbowcond_eq)

    sg_cos, sg_cond = process.get_w2v(args.sgcos_pickle,
                                      args.sgcond_pickle,
                                      norms, args.sg_path, False,
                                      args.sgcond_eq)

    tsg_pos = process.get_tsg(args.tsgpos_pickle, norms, args.tsg_vocabpath,
                          args.tsgpos_lambdapath, args.tsgpos_gammapath)

    tsg_neg = process.get_tsg(args.tsgneg_pickle, norms, args.tsg_vocabpath,
                              args.tsgneg_lambdapath, args.tsgneg_gammapath,
                              args.tsgneg_mupath)

    tsg_neg_norm = process.get_tsg(args.tsgnegnorm_pickle, norms,
                                   args.tsg_vocabpath, args.tsgneg_lambdapath,
                                   args.tsgneg_gammapath, None)

    tsg_freq = process.get_tsgfreq(args.tsgfreq_pickle, norms,
                                   args.tsg_vocabpath, args.tsg_countspath,
                                   args.tsg_idspath)

    #for cue in tsg_freq:
        #assert len(tsg_freq[cue]) == len(tsg_pos[cue])
    glove_cos, glove_cond =  process.get_glove(args.glovecos_pickle,
                                               args.glovecond_pickle,
                                               args.glove_path, norms)

    # Find the common pairs among the different models
    allpairs = process.get_allpairs(args.allpairs_pickle, norms, cbow_cos,
                                    sg_cos, tsg_pos, glove_cos)
    asympairs = process.get_asym_pairs(norms, allpairs)
    print("common pairs: %d, asym pairs: %d" % (len(allpairs), len(asympairs)))

    commonwords = set(tsg_pos.keys()) & set(cbow_cos.keys()) & \
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
                ("tsg-pos", tsg_pos),
                ("tsg-neg", tsg_neg),
                ("tsg-neg-norm", tsg_neg_norm),
                ("glove-cos", glove_cos),
                ("glove-cond", glove_cond)]

               # ("tsg-freq", tsg_freq),

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
