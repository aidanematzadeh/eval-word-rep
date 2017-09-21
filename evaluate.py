import itertools

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

"""
Calculate the ranked correlation between items that between the two list
of scores.
"""
rank_correlation = scipy.stats.spearmanr


def asymmetry_ratio(scores, cue, target):
    """
    Find the ratio of P(w1|w2) / P(w2|w1) for the cue-target pair.
    """
    if scores[target][cue] == 0:
        return float('inf')
    return scores[cue][target] / scores[target][cue]


def triangle_inequality_threshold(stype, tuples, scores):
    """
    For each method, find the thresholds that produce the same number of pairs
    as used in Griffiths et al.

    Find the pairs such that P(w2|w1) and P(w3|w2) are greater than the thresholds;
    keep the values of P(w3|w1).

    Triangle inequality: P(w2|w1) + P(w3|w2) >= P(w3|w1)

    This is based on Griffiths et al. (2007).

    scores: dictionary in the format of scores[w1][w2] = value
    ratios holds min(P(w2|w1) + P(w3|w2))/P(w3|w1)

    """

    # Creating a list of scores such that we get a similar number of pairs for
    # each method with percentile.
    values = [min(scores[w1][w2], scores[w2][w3]) for w1, w2, w3 in tuples]

    # Finding a threshold such that the number of pairs greater than that
    # threshold are similar to norms. We keep the thresholds on norms similar to
    # Griffiths et al.
    percentiles = [0, 98.5, 99.1, 99.5, 99.8, 99.94, 99.99]
    thresholds = [np.percentile(values, percentile) for percentile in percentiles]

    prob_dist_thresh = {t: [] for t in thresholds}
    ratios = {t: [] for t in thresholds}
    for w1, w2, w3 in tuples:
        for t in thresholds:
            if scores[w1][w2] > t and scores[w2][w3] > t:
                prob_dist_thresh[t].append(scores[w1][w3])
                ratios[t].append(min(scores[w1][w2], scores[w2][w3]) / scores[w1][w3])

    # Keeping the similarity values assigned to each pair in the tuple
    values = [(scores[w1][w2], scores[w2][w3], scores[w1][w3]) for w1, w2, w3 in tuples]
    values = list(itertools.chain(*values))

    return prob_dist_thresh, values, ratios


def plot_triangle_inequality(te_dist, sim_dist, name):
    thresholds = sorted(te_dist.keys())
    xmax = np.max(te_dist[thresholds[0]])
    # plot the triangle inequality distributions
    for normed_flag in (True, False):
        plt.clf()
        plt.subplots_adjust(hspace=1.2)
        for index, thres in enumerate(thresholds):
            ax = plt.subplot(len(te_dist), 1, index+1)
            plt.hist(te_dist[thres], label="%.5f" % (thres), normed=normed_flag)
            ax.set_xlim(xmin=0, xmax=xmax)
            ax.set_title("pairs %d" % len(te_dist[thres]), fontsize=8)
            plt.legend(prop={'size':8})
            if index != len(thresholds) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        plt.savefig("%s_%d.png"% (name, normed_flag))

    # plot the histogram of the similarity values
    plt.clf()
    plt.hist(sim_dist, label=name)
    #ax.set_xlim(xmin=0)
    plt.title('Number of pairs %d' % len(sim_dist))
    plt.legend(prop={'size':8})
    plt.savefig("%s_pairs.png" % name)

def plot_percentile_rank(te_data, filename):
    plt.clf()
    plt.xscale('log')
    plt.gca().invert_xaxis()

    markers = itertools.cycle(('>', '+', '.', 'o', '*'))
    for (stype, te_dist, sim_dist), marker in zip(te_data, markers):
        #if 'tasa' not in stype: continue
        x, y, z = [], [], []
        for _, te in sorted(te_dist.items())[1:]:
            min_value = min(te)
            x.append(len(te))
            y.append(scipy.stats.percentileofscore(sim_dist, min_value, kind='rank'))
            #y.append(scipy.stats.percentileofscore(te, min_value, kind='rank'))
            z.append(min_value)
        print(stype, x, y, z)
        plt.plot(x, y, label=stype, marker=marker, linestyle="-")

    plt.legend()
    plt.savefig(filename)

def sort_pairs(scores, allpairs):
    """ For each key in dictionary, sort the items associated with it """
    sorted_scores = defaultdict(dict)
    for cue, target in allpairs:
        sorted_scores[cue][target] = scores[cue][target]

    sorted_scores = {cue: sorted(cue_scores.items(), key=lambda score: score[1], reverse=True)
                     for cue, cue_scores in sorted_scores.items()}
    return sorted_scores

def sort_all(scores, words, commonwords):
    words = set.intersection(words, commonwords)
    return sort_pairs(scores, itertools.permutations(words, 2))

def median_rank(gold, scores, n):
    """ calculate the median rank of the first n associates """
    ranks = [[] for _ in range(n)]
    maxranks = [[] for _ in range(n)]

    for cue, gold_cue_scores in gold.items():
        cue_scores = scores[cue]
        for index, (gold_score, _) in enumerate(gold_cue_scores[:n]):
            for target_index, (score, _) in enumerate(cue_scores):
                if score == gold_score:
                    ranks[index].append(target_index + 1)
                    break

            maxranks[index].append(len(cue_scores))

    return ranks, maxranks
