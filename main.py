"""
Run the evaluation methods.
"""
import sys
import numpy as np

from evaluate import Evaluation
from process import ProcessData

if __name__ == "__main__":

    # Read file path for different resources
    nelson_path = sys.argv[1]
    google_path = sys.argv[2]
    # Path to the list of words that are used in evaluation
    filter_path = sys.argv[3]

    process = ProcessData(filter_path, google_path, nelson_path, True)
    norms_fsg = process.load_scores(nelson_path)
    word2vec_cond = process.load_scores(google_path)
    word2vec_cos = process.load_scores(filter_path)

    pairs = process.get_pairs(norms_fsg)
    print("number of pairs", len(pairs))

    print("Asymmetries")
    evaluate = Evaluation()
    scores_ratio = {}
    scores_dif = {}
    for stype, scores in [("norms", norms_fsg), ("word2vec-cond", word2vec_cond)]:
        scores_ratio[stype], scores_dif[stype] = evaluate.asymmetry(scores, pairs)

        # Sort the asymmetries based on the ratio
        #sorted_ratio = [(key, value) for (value, key) in sorted(zip(scores_ratio.values(), scores_ratio.keys()), reverse=True)]
        #print(stype, sorted_ratio[:19])
    #
    rho_ratio = evaluate.rank_correlation(scores_ratio["norms"], scores_ratio["word2vec-cond"])
    print("Spearman of asymmetries based ratio ", rho_ratio)
    #
    rho_dif = evaluate.rank_correlation(scores_dif["norms"], scores_dif["word2vec-cond"])
    print("Spearman of asymmetries based difference ", rho_dif)

    mismatch = 0
    for index in range(len(pairs)):
        if scores_ratio["norms"][index] > 30 and \
                (scores_ratio["word2vec-cond"][index] < 1):
                    print(pairs[index], scores_ratio["norms"][index], scores_ratio["word2vec-cond"][index])
                    mismatch += 1
    print("mismatch", mismatch)


    #
    tuples = process.get_tuples(norms_fsg)
    print("Number of tuples", len(tuples))
    #
    # Examine whether the traingle inequality holds in Nelson norms
    thresholds = np.arange(0.05, 0.65, 0.1)
    norm_prob_dist_thresh, norm_dif, norm_ratio = evaluate.traingle_inequality_threshold(tuples, norms_fsg, thresholds)
    evaluate.plot_traingle_inequality(norm_prob_dist_thresh, "nelson_")

    # Examine whether the triangle inequlaity holds for word2vec using cosine
    # and conditional probability
    w2vcos_prob_dist_thresh, w2vcos_dif, w2vcos_ratio = evaluate.traingle_inequality_threshold(tuples, word2vec_cos, thresholds)
    evaluate.plot_traingle_inequality(w2vcos_prob_dist_thresh, "word2vec_cos")

    w2vcond_prob_dist_thresh, w2vcond_dif, w2vcond_ratio = evaluate.traingle_inequality_threshold(tuples, \
    word2vec_cond, np.arange(0.00035, 0.00045, 0.0002))#np.arange(0.01, 0.3, 0.05))
    evaluate.plot_traingle_inequality(w2vcond_prob_dist_thresh, "word2vec_cond")

    rho_te = evaluate.rank_correlation(norm_dif, w2vcond_dif)
    print("correlation in traingle ineq cond prob", rho_te)
    for index in range(len(tuples)):
        if norm_ratio[index] > 30 and \
                w2vcond_ratio[index] < 1:
                    print(tuples[index], norm_ratio[index], w2vcond_ratio[index])
    #
    rho_te = evaluate.rank_correlation(norm_dif, w2vcos_dif)
    print("correlation in traingle ineq cosine", rho_te)
    for index in range(len(tuples)):
        if norm_ratio[index] > 30 and \
                w2vcos_ratio[index] < 1:
                    print(tuples[index], norm_ratio[index], w2vcond_ratio[index])

    exit()

    # Median rank of associates
    # Sort the norm associates
    gold_associates = evaluate.sort_scores(norms_fsg)

    # Sort the word2vec asscociates
    word2vec_cos_sorted = evaluate.sort_scores(word2vec_cos)
    word2vec_logcond_sorted = evaluate.sort_scores(word2vec_cond)


    """
    for cue in word2vec_cos_sorted:
        print("cos", cue, word2vec_cos_sorted[cue][:10])
        print("logcond", cue, word2vec_logcond_sorted[cue][:10])
        print("norm", cue, gold_associates[cue][:10])
    """
    print("Word2Vec, cosine")
    w2v_cos_ranks = evaluate.calc_median_rank(gold_associates, word2vec_cos_sorted, 10)

    print("\nWord2Vec, conditional prob")
    w2v_logcond_ranks = evaluate.calc_median_rank(gold_associates, word2vec_logcond_sorted, 10)










