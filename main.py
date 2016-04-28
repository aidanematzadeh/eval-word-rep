"""
Run the evaluation methods.
"""
import time
import sys
import numpy as np

from evaluate import Evaluation
from process import ProcessData
import constants as const

if __name__ == "__main__":
    
    # Read file path for different resources
    nelson_norms = sys.argv[1]
    google_path = sys.argv[2]
    # Path to the list of words that are used in evaluation
    filter_path = sys.argv[3]
    
    # Read the word list that the evaluation is on. These words should occur in
    # both Nelson norms and Google vectors
    process = ProcessData()
    assoc_list = process.read_filter_words(filter_path)
    print("assoc list", len(assoc_list))

    word2vec_model = process.load_word2vec_model(google_path)
    
    word_list = []
    for word in assoc_list:
        if word in word2vec_model:
            word_list.append(word)
        else:
            print("Word not in word2vec", word)
    print("word list", len(word_list))


    norms_fsg = process.read_norms(nelson_norms, word_list)
    print("norm list", len(norms_fsg))
    
    tuples = process.find_norm_pairs(norms_fsg)
    print("Number of tuples in data", len(tuples))
    word2vec_cos, word2vec_logcond = process.read_word2vec(word2vec_model, norms_fsg, word_list) 
    
    print("Calculating asymmetries")
    evaluate = Evaluation()
    for stype, scores in [("Norms", norms_fsg), ("Word2Vec Cond", word2vec_logcond)]:
        scores_ratio, scores_dif = evaluate.find_asymmetry(scores)
    
        # Sort the asymmetries based on the ratio
        sorted_ratio = [(key, value) for (value, key) in sorted(zip(scores_ratio.values(), scores_ratio.keys()), reverse=True)]
        print(stype, sorted_ratio[:10])

        #plt.hist(asym_ratio.values())
        #fig = plt.gcf()
        #plt.show()


    # Examine whether the traingle inequality holds in Nelson norms
    thresholds = np.arange(0.05, 0.65, 0.1)
    norm_prob_dist_thresh, norm_differences = evaluate.traingle_inequality_threshold(tuples, norms_fsg, thresholds)
    evaluate.plot_traingle_inequality(norm_prob_dist_thresh, "nelson_")
    
    # Examine whether the triangle inequlaity holds for word2vec using cosine
    # and conditional probability
    w2vcos_prob_dist_thresh, w2vcos_differences = evaluate.traingle_inequality_threshold(tuples, word2vec_cos, thresholds)
    evaluate.plot_traingle_inequality(w2vcos_prob_dist_thresh, "word2vec_cos")

    w2vlogcond_prob_dist_thresh, w2vlogcond_differences = evaluate.traingle_inequality_threshold(tuples, \
    word2vec_logcond, np.arange(0.00035, 0.00045, 0.0002))#np.arange(0.01, 0.3, 0.05))
    
    #print("-----", w2vlogcond_prob_dist_thresh)
    evaluate.plot_traingle_inequality(w2vlogcond_prob_dist_thresh, "word2vec_logcond")
    
    
    # Median rank of associates
    # Sort the norm associates
    gold_associates = evaluate.sort_scores(norms_fsg)

    # Sort the word2vec asscociates
    word2vec_cos_sorted = evaluate.sort_scores(word2vec_cos)
    word2vec_logcond_sorted = evaluate.sort_scores(word2vec_logcond)

    
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










