import sys
import os
import constants as const
import itertools

class ProcessData:
    """ This class reads and process data used for evaluation or training.
    """
    
    def __init__(self):
        pass 

    def read_norms(self, norms_dir):
        """ Read Nelson norms for the evaluation methods.
        Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
        """
        norms = {}
        for filename in os.listdir(norms_dir):
            norm_file = open(norms_dir + "/" + filename, 'r')
            norm_file.readline()
            
            for line in norm_file:
                nodes = line.strip().split(',')
                
                cue = nodes[0].strip().lower() 
                target = nodes[1].strip().lower() 
                
                if not norms.has_key(cue):
                    norms[cue] = {}
                if not norms[cue].has_key(target):
                    norms[cue][target] = {}
                
                norms[cue][target][const.FSG] = float(nodes[5])
                
                # Mark the probabilities that do not exist with 0
                try:
                    norms[cue][target][const.BSG] = float(nodes[6])
                except ValueError:
                    norms[cue][target][const.BSG] = 0

        return norms

    def find_norm_pairs(self, norms):
        """ Find all the three words for which P(w2|w1), P(w3|w2), and P(w3|w1) exist 
        This is equivalent to the subset of the combination of 3-length ordered tuples 
        that their pairs exist in Nelson norms.
        """
        tuples = []

        for w1 in norms:
            for w2 in norms[w1]:
                if not norms.has_key(w2):
                    continue

                for w3 in norms[w2]:
                    if norms[w1].has_key(w3):
                        tuples.append((w1, w2, w3))
        return tuples
         




