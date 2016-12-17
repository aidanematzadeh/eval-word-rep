import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
import numpy


def rank_correlation(list1, list2):
    """
    Calculate the ranked correlation between items that between the two list of scores.
    """
    return scipy.stats.spearmanr(list1, list2)

def asymmetry(scores, pairs):
    """
    Find the ratio of p(w1|w2)/p(w2|w1) and p(w1|w2)-p(w2|w1) for each pair in pairs.
    """
    ratios = []
    differences = []
    for cue, target in pairs:
        differences.append(scores[cue][target] - scores[target][cue])
        try:
            ratios.append(scores[cue][target]  / scores[target][cue])
        except ZeroDivisionError:
            print(ZeroDivisionError)
            ratios.append(float('inf'))
    return ratios, differences

#TODO
def traingle_inequality_threshold(tuples, scores, common_words, thresholds=None):
    """
    Find the pairs such that P(w2|w1) and P(w3|w2) are greater than the threshold;
    plot the distribution of P(w3|w1).

    Traingle inequaliy: P(w2|w1) + P(w3|w2) >= P(w3|w1)

    This is based on Griffiths et al. (2007).

    scores: dictionary in the format of scores[w1][w2] = value
    differences holds P(w2|w1) + P(w3|w2) - P(w3|w1)

    """
    values = []
    for cue in scores:
        if cue in common_words:
            values += list(scores[cue].values())

    thresholds = [numpy.percentile(values, 97)]
    thresholds.append(numpy.percentile(values, 95))
    thresholds.append(numpy.percentile(values, 93))
    thresholds.append(numpy.percentile(values, 91))
    #print("------", thresholds)

    prob_dist_thresh = {}
    differences, ratios = [] , []
    for t in thresholds:
        prob_dist_thresh[t] = []

    for w1,w2,w3 in tuples:
        # TODO: this can be done more efficiently, not need to do this for all the thresholds
        for t in thresholds:
            if scores[w1][w2] >= t and scores[w2][w3] >= t:
                prob_dist_thresh[t].append(scores[w1][w3])

        differences.append(min(scores[w1][w2], scores[w2][w3]) - scores[w1][w3])
        ratios.append(min(scores[w1][w2], scores[w2][w3]) / scores[w1][w3])

        #print((w1,w2,w3), differences[-1])

    return prob_dist_thresh, ratios, differences


def plot_traingle_inequality(dist, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patterns = ('', '+', '*', '\\', '*', 'o', 'O', '.')

    #colormapname = "YlGnBu"
    index = 0
    for thres in sorted(dist.keys()):
        #print(name, thres, len(dist[thres]))
        ax.hist(dist[thres], label="%.5f" %(thres), hatch = patterns[index])#, color=plt.get_cmap(colormapname))
        index +=1
    #ax.set_ylim(0, 10)
    ax.set_xlim(xmin=0)
    ax.legend()
    fig.savefig(name.replace('.',''))
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    index = 0
    for thres in sorted(dist.keys()):
        ax.hist(dist[thres], label="%0.5f"% (thres), normed=True, hatch= patterns[index])#, color=plt.get_cmap(colormapname))
        index += 1
    ax.set_xlim(xmin=0)
    ax.legend()
    fig.savefig(name.replace('.','') + "_normed.png")


def sort_pairs(scores, allpairs):
    """ For each key in dictionary, sort the items associated with it """
    sorted_scores = {}
    for cue, target in allpairs:
        if not cue in sorted_scores:
            sorted_scores[cue] = {}
        sorted_scores[cue][target] = scores[cue][target]

    for cue in sorted_scores.keys():
        sorted_scores[cue] = sorted(sorted_scores[cue].items(), key=operator.itemgetter(1), reverse=True)

    return sorted_scores

def sort_all(scores, norms, commonwords):
    sorted_scores = {}
    for cue1 in norms.keys():
        if not (cue1 in commonwords):
            continue
        for cue2 in norms.keys():
            if not (cue2 in commonwords):continue
            if cue1==cue2: continue
            if not cue1 in sorted_scores:
                sorted_scores[cue1] = {}
            sorted_scores[cue1][cue2] = scores[cue1][cue2]

    for cue in sorted_scores.keys():
        sorted_scores[cue] = sorted(sorted_scores[cue].items(), key=operator.itemgetter(1), reverse=True)
    return sorted_scores


def median_rank(gold, scores,  n=3):
    """ calculate the median rank of the first n associates """
    ranks, maxranks = {} , {}
    for r in range(n):
        ranks[r] = []
        maxranks[r] = []

    for cue in gold:
        for index in range(min(len(gold[cue]), n)):
            target = gold[cue][index][0]

            for j in range(len(scores[cue])):
                if scores[cue][j][0] == target:
                    ranks[index].append(j+1)
                    break

            maxranks[index].append(len(scores[cue]))
    return ranks, maxranks




