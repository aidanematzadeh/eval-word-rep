import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
import numpy
import itertools

def rank_correlation(list1, list2):
    """
    Calculate the ranked correlation between items that between the two list of
    scores.
    """
    return scipy.stats.spearmanr(list1, list2)


def asymmetry(scores, pairs):
    """
    Find the ratio of p(w1|w2)/p(w2|w1) and p(w1|w2)-p(w2|w1) for each pair
    in pairs.
    """
    ratios = []
    differences = []
    for cue, target in pairs:
        differences.append(scores[cue][target] - scores[target][cue])
        try:
            ratios.append(scores[cue][target] / scores[target][cue])
        except ZeroDivisionError:
            print(ZeroDivisionError)
            ratios.append(float('inf'))
    return ratios, differences


def traingle_inequality_threshold(stype, tuples, scores):
    """
    For each method, find the thresholds that produce the same number of pairs
    as used in Griffiths et al.

    Find the pairs such that P(w2|w1) and P(w3|w2) are greater than the thresholds;
    keep the values of P(w3|w1).

    Traingle inequaliy: P(w2|w1) + P(w3|w2) >= P(w3|w1)

    This is based on Griffiths et al. (2007).

    scores: dictionary in the format of scores[w1][w2] = value
    ratios holds min(P(w2|w1) + P(w3|w2))/P(w3|w1)

    """

    # Creating a list of scores such that we get a similar number of pairs for
    # each method with percentile.
    values = []
    for w1, w2, w3 in tuples:
        values.append(min(scores[w1][w2], scores[w2][w3]))

    # Finding a threshold such that the number of pairs greater than that
    # threshold are similar to norms. We keep the thresholds on norms similar to
    # Griffiths et al.
    thresholds = []
    thresholds.append(numpy.percentile(values, 99.99))
    thresholds.append(numpy.percentile(values, 99.94))
    thresholds.append(numpy.percentile(values, 99.8))
    thresholds.append(numpy.percentile(values, 99.5))
    thresholds.append(numpy.percentile(values, 99.1))
    thresholds.append(numpy.percentile(values, 98.5))
    thresholds.append(numpy.min(values))
    thresholds = sorted(thresholds)

    prob_dist_thresh, ratios = {} , {}
    for t in thresholds:
        prob_dist_thresh[t], ratios[t] = [], []
    for w1, w2, w3 in tuples:
        for t in thresholds:
            if scores[w1][w2] > t and scores[w2][w3] > t:
                prob_dist_thresh[t].append(scores[w1][w3])
                ratios[t].append(min(scores[w1][w2], scores[w2][w3]) / scores[w1][w3])

    # Keeping the similarity values assigned to each pair in the tuple
    values = []
    for w1, w2, w3 in tuples:
        values.append(scores[w1][w2])
        values.append(scores[w2][w3])
        values.append(scores[w1][w3])

    return prob_dist_thresh, values, ratios


def plot_traingle_inequality(te_dist, sim_dist, name):
    n = len(te_dist.keys())

    thresholds = sorted(te_dist.keys())
    xmax = numpy.max(te_dist[thresholds[0]])
    # plot the triangle inequality distributions
    for normed_flag in (True, False):
        plt.clf()
        plt.subplots_adjust(hspace=1.2)
        for index, thres in enumerate(thresholds):
            ax = plt.subplot(n, 1, index+1)
            plt.hist(te_dist[thres], label="%.5f" % (thres), normed=normed_flag)
            ax.set_xlim(xmin=0)
            ax.set_xlim(xmax=xmax)
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

    marker = itertools.cycle(('>', '+', '.', 'o', '*'))
    for stype, te_dist, sim_dist in te_data:
        #if 'tasa' not in stype: continue
        x, y ,z = [], [], []
        for t in sorted(te_dist.keys())[1:]:
            min_value = numpy.min(te_dist[t])
            x.append(len(te_dist[t]))
            y.append(scipy.stats.percentileofscore(sim_dist, min_value, kind='rank'))
            #y.append(scipy.stats.percentileofscore(te_dist[t], min_value, kind='rank'))
            z.append(min_value)
        print(stype, x, y, z)
        plt.plot(x,y, label=stype, marker=next(marker), linestyle="-")

    plt.legend()
    plt.savefig(filename)

def sort_pairs(scores, allpairs):
    """ For each key in dictionary, sort the items associated with it """
    sorted_scores = {}
    for cue, target in allpairs:
        if not cue in sorted_scores:
            sorted_scores[cue] = {}
        sorted_scores[cue][target] = scores[cue][target]

    for cue in sorted_scores.keys():
        sorted_scores[cue] = sorted(sorted_scores[cue].items(), key=operator.itemgetter(1,0), reverse=True)

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
        sorted_scores[cue] = sorted(sorted_scores[cue].items(), key=operator.itemgetter(1,0), reverse=True)
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




