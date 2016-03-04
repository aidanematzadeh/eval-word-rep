"""
Load a word2vec model and run the evaluation methods.
"""

import gensim
import time
import sys
import matplotlib.pyplot as plt


from evaluate import Evaluation
from process import ProcessData
import constants as const

def eval_asym():
    pass


if __name__ == "__main__":
    norm_dir = sys.argv[1]

    process_data = ProcessData()
    norms = process_data.read_norms(norm_dir)
    tuples = process_data.find_norm_pairs(norms)
    print "Number of tuples ", len(tuples)
    print "Done with Nelson norms."

    evaluate = Evaluation()
    
    norm_prob_dist_thresh, norm_differences = evaluate.traingle_inequality_threshold(tuples, norms, const.FSG, [0.15, 0.55])
    filename = "nelson_"
    for t in norm_prob_dist_thresh:
        print len(norm_prob_dist_thresh[t])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.hist(norm_prob_dist_thresh[t])
        ax.set_xlim([0, 0.8])
        plt.savefig(filename+ str(t) + ".png")
   
    print "done"
    
    model = gensim.models.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
   
    print "done loading the model"
    word2vec_rep = {}
    for cue in norms:
        word2vec_rep[cue] = {}
        for target in norms:
            word2vec_rep[cue][target] = {}
            try:
                word2vec_rep[cue][target][const.COS] = model.similarity(cue, target)
            except RuntimeError:
                print cue, target
    print "done calculating similarities"

    w2v_prob_dist_thresh, w2v_differences = evaluate.traingle_inequality_threshold(tuples, word2vec_rep, const.cos, [0.15, 0.55])
    filename = "word2vec_"
    for t in w2v_prob_dist_thresh:
        print len(norm_prob_dist_thresh[t])
        fig = plt.gcf()
        ax = fig.add_subplot(1, 1, 1)
        plt.hist(w2v_prob_dist_thresh[t])
        plt.set_xlim([0, 0.8])
        plt.savefig(filename + str(t) + ".png")


    


'''
asym_ratio, asym_dif = Evaluation().find_asymmetry(norms)
# Sort the asymmetries based on the ratio
sorted_asym_ratio = [(key, value) for (value, key) in sorted(zip(asym_ratio.values(), asym_ratio.keys()), reverse=True)]
print sorted_asym_ratio[:10]

plt.hist(asym_ratio.values())
#plt.title("Gaussian Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")

fig = plt.gcf()
plt.show()
'''



print "------------------------"
#instances = Evaluation().triangle_inequality(norms)
#sorted_instances = [(key, value) for (value, key) in sorted(zip(instances.values(), instances.keys()), reverse=True)]
#print sorted_instances[:9]







