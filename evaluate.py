import numpy

class Evaluation:
    """
    This class implements methods for evulating word representations.
        
    Members:
        Nelson norms --




    """
    def __init__(norms, norms_dir):
        
        if not norms:
            self._norms = ProcessNorms.read_norms(norms_dir)


    # How did they treat the words for which p(w1|w2) did not exist?
    def find_asymmetry(norms):
        """
        find the ratio of p(w1|w2)/p(w2|w1) for pairs that both probabilities exist.
        """
        no_prob = 0 # counting the pairs for which one of the probabilities does not exist
        ratios = {}

        for cue in norms:
            for target in norms:
                if norms[cue][target][constant.FSG] == -1 or norms[cue][target][constant.BSG] == -1:
                    no_porb += 1
                    continue
                if (cue, target) in ratios.keys() or (target, cue) in ratios.keys():
                    continue

                ration[(cue, target)] =   norms[cue][target][constant.FSG] / norms[cue][target][constant.BSG] 
                print cue, target,  norms[cue][target][constant.FSG] / norms[cue][target][constant.BSG] 
        
        return ratios

