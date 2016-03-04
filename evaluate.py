import numpy
import constants as const
import math

import matplotlib.pyplot as plt


class Evaluation:
    """
    This class implements methods for evulating word representations.
        
    Members:
        Nelson norms --




    """
    
    def __init__(self):
        pass
    #    
    #    if not norms:
    #        self._norms = ProcessNorms.read_norms(norms_dir)

    
#    def     

    #spearman = scipy.stats.spearmanr(gold_pairs, graph_pairs)

    def find_asymmetry(self, norms):
        """
        Find the ratio of p(w1|w2)/p(w2|w1) for pairs that both probabilities exist (is number) 
        and at least one of the probabilities is great than zero.
        """
        no_prob = 0 # counting the pairs for which one of the probabilities does not exist
        all_pairs = 0
        ratios = {}
        differences = {}

        not_cues = set([])

        for cue in norms:
            for target in norms[cue]:
                all_pairs += 1

                if not target in norms.keys():
                    no_prob +=1
                    not_cues.add(target)
                    continue

                if (cue, target) in ratios.keys() or (target, cue) in ratios.keys():
                    continue
                
                differences[(cue, target)] = math.fabs(norms[cue][target][const.FSG] - norms[cue][target][const.BSG])
                try:
                    ratios[(cue, target)] = max(norms[cue][target][const.FSG], norms[cue][target][const.BSG]) / min(norms[cue][target][const.FSG], norms[cue][target][const.BSG])
                except ZeroDivisionError:
                    ratios[(cue, target)] = float('inf')
       
        print "no prob", no_prob, "all pairs", all_pairs, "asymmetries", len(ratios.keys())
        print "Not used as a cue ", len(not_cues)
        
        return ratios, differences

    
    def traingle_inequality(self, norms, which_function):
        """
        Calls the right function depending on the type of analysis 
        """

        if which_function == const.NOTTE:
            return self._not_traingle_inequality(norms)
        elif which_function == const.GRTE:
            pass


    def _not_triangle_inequality(self, norms):
        """
        Find pairs for which the triangle inequality does not hold, that is, d(z,z) > d(x,y) + d(y,z). 
        Shows that to what extent we can treat the similarities as a distance metric.
        """

        return 
    
    def traingle_inequality_threshold(self, tuples, scores, score_type, thresholds):
        """
        Find the pairs such that P(w2|w1) and P(w3|w2) are greater than the threshold;
        plot the distribution of P(w3|w1).

        Traingle inequaliy: P(w2|w1) + P(w3|w2) >= P(w3|w1)

        This is based on Griffiths et al. (2007).

        scores: dictionary in the format of scores[w1][w2][score_type] = value
        differences holds P(w2|w1) + P(w3|w2) - P(w3|w1)

        """
        prob_dist_thresh = {}  
        differences = {}
        for t in thresholds:
            prob_dist_thresh[t] = []
        
        for w1,w2,w3 in tuples:
            # TODO: this can be done more efficiently, not need to do this for all the thresholds
            for t in thresholds:
                if scores[w1][w2][score_type] >= t and scores[w2][w3][score_type] >= t:
                    prob_dist_thresh[t].append(scores[w1][w3][score_type])
            differences[(w1, w2, w3)] = scores[w1][w2][score_type] + scores[w2][w3][score_type] - scores[w1][w3][score_type]
        
        return prob_dist_thresh, differences














