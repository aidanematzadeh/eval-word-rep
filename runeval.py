"""
Run the evaluation methods.
"""
#import sys
import numpy as np
import argparse
from collections import defaultdict
from evaluate import Evaluation
from process import ProcessData

def defaultdict_list():
    return defaultdict(list)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("nelson", type=str, help="Input Neslon norms path or pickle file.")
    argparser.add_argument("word2vec", type=str, help="Input word2vec path or pickle file.")
    argparser.add_argument("lda", type=str, help="the LDA model")
    argparser.add_argument("ldawords", type=str, help="the LDA words")
    argparser.add_argument("outdir", default='', help="Directory to place output files. (default='')")
    args = argparser.parse_args()

    process = ProcessData(args.nelson, args.word2vec, args.ldawords, args.outdir)
    norms_fsg = process.norms_fsg
    word2vec_cond = process.word2vec_cond
    word2vec_cos = process.word2vec_cos
    common_words = process.common_words
    lda = process.read_lda(args.lda, norms_fsg, common_words)

    pairs = process.get_pairs(norms_fsg, common_words)
    print("number of pairs", len(pairs))

    evaluate = Evaluation()

    print("Asymmetries")
    asyms= defaultdict(defaultdict_list)
    evallist = [("norms", norms_fsg), ("word2vec-cond", word2vec_cond), \
            ("word2vec-cos", word2vec_cos), ("lda", lda)]
    for stype, scores in evallist:
        if stype == "word2vec-cos": continue
        asyms["ratio"][stype], asyms["difference"][stype] = evaluate.asymmetry(scores, pairs)

        # Sort the asymmetries based on the ratio
        #sorted_ratio = [(key, value) for (value, key) in sorted(zip(scores_ratio.values(), scores_ratio.keys()), reverse=True)]
        #print(stype, sorted_ratio[:19])

    for b in asyms:
        for stype in asyms[b]:
            rho = evaluate.rank_correlation(asyms[b]["norms"], asyms[b][stype])
            print("correlation between norms and %s (%s of asymmetries): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))

            if stype == "difference": continue
            for index in range(len(pairs)):
                if asyms[b]["norms"][index] > 30 and \
                    (asyms[b][stype][index] < 1):
                        print(pairs[index], asyms[b]["norms"][index], asyms[b][stype][index])
            print()

    asym = None
    tuples = process.get_tuples(norms_fsg, lda)
    print("Number of tuples", len(tuples))
    #
    print("Triangle Inequality")
    # Examine whether the traingle inequality holds
    thresholds = {}
    thresholds["norms"] = np.arange(0.45, 0.85, 0.1)
    thresholds["word2vec-cos"] = np.arange(0.75, 1, 0.1)
    thresholds["word2vec-cond"] =  np.arange(0.0002, 0.00035, 0.00005)#np.arange(0.00035, 0.00051, 0.00005)
    thresholds["lda"] = np.arange(0.005, 0.025, 0.005)
    te = defaultdict(defaultdict_list)
    for stype, scores in evallist:
        te_dist, te["ratios"][stype], te["differences"][stype] = evaluate.traingle_inequality_threshold(tuples, scores, thresholds[stype])
        evaluate.plot_traingle_inequality(te_dist, process.outdir + stype + "_")

    for b in te:
        for stype in te[b]:
            rho = evaluate.rank_correlation(te[b]["norms"], te[b][stype])
            print("correlation between norms and %s (%s of traingle inequality): (%.2f, %.2f)" % (stype, b, rho[0], rho[1]))

            if stype == "difference": continue
            for index in range(len(tuples)):
                if te[b]["norms"][index] > 30 and \
                        te[b][stype][index] < 1:
                            print(tuples[index], te[b]["norms"][index], te[b][stype][index])


    print("Associations")
    # Median rank of associates
    # Sort the norm associates
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









