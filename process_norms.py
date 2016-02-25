import sys
import os

class ProcessNorms:
    """
    This class reads Nelson norms and process them for the evaluation methods.
    """

    def read_norms(norms_dir):
    #CUE, TARGET, NORMED?, #G, #P, FSG, BSG,

    norms = {}
#    norms_dic = {} # [cue][target] = prob
    for filename in os.listdir(norms_dir):
        norm_file = open(norms_dir + "/" + filename, 'r')
        for line in norm_file:
            if line.startswith("<"):
                continue

            nodes = line.strip().split(',')
            cue = nodes[0].strip().lower() + ":N"
            target = nodes[1].strip().lower() + ":N"

            if not norms.has_key(cue):
                norms[cue] = {}
            if not norms.has_key(norms[cue][target]):
                norms[cue][target] = {}
            
            try:
                norms[cue][target][Constants.FSG] = float(nodes[5])
            except ValueError:
                norms[cue][target][Constants.FSG] = -1
            
            try:
                norms[cue][target][Constants.BSG] = float(nodes[6])
            except ValueError:
                norms[cue][target][Constants.BSG] = -1



    return norms

    def find_associates(norms):

