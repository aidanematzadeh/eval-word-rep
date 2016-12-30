import sys
import numpy as n

import matplotlib
import matplotlib.pyplot as plt

import dtsg
import hofonlineldavb as poslda
import process

matplotlib.use('Agg')


def read_data(datapath, normsid=None):
    # Reading the word ids and counts
    idfile = open(datapath + "_ids", 'r')
    countfile = open(datapath + "_counts", 'r')

    ids, counts = [], []
    for index, (idline, ctline) in enumerate(zip(idfile, countfile)):
        # if normsid != None and not index in normsid: continue
        docids, doccounts = [], []
        for wid, count in zip(idline.split(), ctline.split()):
            docids.append(int(wid))
            doccounts.append(int(count))
        ids.append(docids)
        counts.append(doccounts)
    return ids, counts


def read_dic(filename):
    word2id, id2word = {}, {}
    wordfreq = []
    with open(filename, 'r') as f:
        for line in f:
            w, wid, wfreq = line.split()
            word2id[w] = int(wid)
            id2word[int(wid)] = w
            wordfreq.append([w, int(wid), int(wfreq)])
    wordfreq.sort(key=lambda tup: tup[2], reverse=True)
    return word2id, id2word, wordfreq


# Main

# infile = sys.argv[1]
# K = int(sys.argv[2])
# alpha = float(sys.argv[3])
# eta = float(sys.argv[4])
# kappa = float(sys.argv[5])
# S = int(sys.argv[6])

datapath = sys.argv[1]
outpath = sys.argv[2]
# To use negative examples or not
negative_flag = sys.argv[3]
# To do batch or online processing
batch_flag = sys.argv[4]
# To train on all the vocabulary or only norms
norm_flag = sys.argv[5]
norms_path = sys.argv[6]

negative_flag = (negative_flag.lower() == 'neg')
batch_flag = (batch_flag.lower() == 'batch')
norm_flag = (norm_flag.lower() == 'norm')

print("negative", negative_flag, "batch", batch_flag, "norm", norm_flag)
vocab2id, id2vocab, wordfreq = read_dic(datapath + "word2id")
vocab_size = len(vocab2id)

normsid = None
if norm_flag:
    normsid = []
    norms = process.get_norms(outpath+"/norms.pickle", norms_path)
    for cue in norms:
        if cue in vocab2id.keys():
            normsid.append(vocab2id[cue])
    print(len(normsid), normsid[1:10])


pos_wordids, pos_wordcts = read_data(datapath + "positive")
assert(len(pos_wordids) == len(pos_wordcts))
assert(len(pos_wordids[0]) == len(pos_wordcts[0]))

# Number of documents
doc_num = len(pos_wordids)
print("number of documents %d" % doc_num)

# assert(len(neg_wordids) == len(neg_wordcts))
# assert(len(neg_wordids[0]) == len(neg_wordcts[0]))
# assert(len(pos_wordids) == len(neg_wordcts))

batch_size = 512
topic_num = 80

tau0 = 1
kappa = 0.5

if batch_flag:
    kappa = 0

if negative_flag:
    print("using negative examples")
    neg_wordids, neg_wordcts = read_data(datapath + "negative")
    model = dtsg.OnlineLDA(vocab2id, K=topic_num, D=doc_num, alpha=0.1,
                           eta=0.00000001, zeta=1, tau0=tau0, kappa=kappa)
else:
    model = poslda.OnlineLDA(vocab2id, K=topic_num, D=doc_num, alpha=0.1,
                             eta=0.01, tau0=tau0, kappa=kappa)

gamma = n.zeros((doc_num, topic_num))
bounds = []
perplexity = []

if batch_flag:
    relative_change = 1
    for counter in range(0, 4):
        gamma, bound = model.update_lambda(pos_wordids, pos_wordcts)
        counts_sum = sum(map(sum, pos_wordcts))
        wbound = bound * len(pos_wordids) / (doc_num * counts_sum)

        if len(bounds) > 1:
            relative_change = bound - bounds[-1]

        print("counter", counter)
        print("relative_change", relative_change)
        print('rho_t = %f,  held-out perplexity estimate = %f, \
              approx bound = %.5f' % (model._rhot, n.exp(-wbound), bound))
        perplexity.append(wbound)
        bounds.append(bound)
else:
    for counter in range(0, 10):
        print("counter", counter)
        i = 0
        while i < doc_num:
            ni = min(i + batch_size, doc_num)
            print(i, ni)
            if negative_flag:
                gamma[i:ni], bound = model.update_lambda(pos_wordids[i:ni],
                                                         pos_wordcts[i:ni],
                                                         neg_wordids[i:ni],
                                                         neg_wordcts[i:ni],
                                                         i, ni)
                counts_sum = (sum(map(sum, pos_wordcts[i:ni])) +
                              sum(map(sum, neg_wordcts[i:ni])))
            else:
                gamma[i:ni], bound = model.update_lambda(pos_wordids[i:ni],
                                                         pos_wordcts[i:ni],
                                                         i, ni)
                counts_sum = sum(map(sum, pos_wordcts[i:ni]))

            wbound = bound * len(pos_wordids[i:ni]) / (doc_num * counts_sum)
            print('%d:  rho_t = %f,  held-out perplexity estimate = %f, approx\
                  bound = %.5f' % (i, model._rhot, n.exp(-wbound), bound))
            perplexity.append(wbound)
            bounds.append(bound)
            i = ni
            print("number of documents processed: %d" % i)
        n.savetxt(outpath + 'gamma%d' % counter, model._gamma)

# Saving the parameters
if negative_flag:
    n.savetxt(outpath + '/mu%d' % len(model._mu.T), model._mu)
n.savetxt(outpath + '/lambda%d' % len(model._lambda.T), model._lambda)
n.savetxt(outpath + '/gamma%d' % len(gamma), gamma)

# Plotting bound and perplexity
plt.plot(bounds)
plt.savefig(outpath + "/bound_plot%d.png" % len(gamma))

plt.plot(perplexity)
plt.savefig(outpath + "/perplexity_plot%d.png" % len(gamma))

# Printing the topics
final_lambda = model._lambda
for k in range(0, len(final_lambda)):
    lambdak = final_lambda[k]
    if negative_flag:
        denom = (lambdak + model._mu[k])
        lambdak = lambdak / denom
    else:
        lambdak = lambdak / sum(lambdak)
    temp = zip(lambdak, range(0, len(lambdak)))
    temp = sorted(temp, key=lambda x: x[0], reverse=True)
    print('topic %d:' % (k))
    # feel free to change the "53" here to whatever fits your screen nicely.
    for index in range(0, 35):
        print('%20s  \t--\t  %.4f' % (id2vocab[temp[index][1]], temp[index][0]))
    print()
