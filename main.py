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
    thresholds = np.arange(0.45, 0.85, 0.1)
    norm_prob_dist_thresh, norm_dif, norm_ratio = evaluate.traingle_inequality_threshold(tuples, norms_fsg, thresholds)
    evaluate.plot_traingle_inequality(norm_prob_dist_thresh, "nelson_")


    # Examine whether the triangle inequlaity holds for word2vec using cosine
    # and conditional probability
    thresholds = np.arange(0.65, 1, 0.1)
    w2vcos_prob_dist_thresh, w2vcos_dif, w2vcos_ratio = evaluate.traingle_inequality_threshold(tuples, word2vec_cos, thresholds)
    evaluate.plot_traingle_inequality(w2vcos_prob_dist_thresh, "word2vec_cos")


    thresholds = np.arange(0.00035, 0.00051, 0.00005)
    w2vcond_prob_dist_thresh, w2vcond_dif, w2vcond_ratio = evaluate.traingle_inequality_threshold(tuples, \
    word2vec_cond, thresholds)
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


    # Median rank of associates
    # Sort the norm associates
    gold_associates = evaluate.sort_scores(norms_fsg)

    # Sort the word2vec asscociates
    w2vcos_sorted = evaluate.sort_scores(word2vec_cos)
    w2vcond_sorted = evaluate.sort_scores(word2vec_cond)


    #for cue in w2vcos_sorted:
    #    print("cos", cue, w2vcos_sorted[cue][:10])
    #    print("logcond", cue, w2vcond_sorted[cue][:10])
    #    print("norm", cue, gold_associates[cue][:10])

    print("Word2Vec, cosine")
    w2vcos_ranks = evaluate.median_rank(gold_associates, w2vcos_sorted, 10)
    for rank in w2vcos_ranks:
        print("median rank associate %d: %.2f", (rank+1, np.median(w2vcos_ranks[rank])))

    print("\nWord2Vec, conditional prob")
    w2vcond_ranks = evaluate.median_rank(gold_associates, w2vcond_sorted, 10)
    for rank in w2vcond_ranks:
        print("median rank associate %d: %.2f", (rank+1, np.median(w2vcond_ranks[rank])))










