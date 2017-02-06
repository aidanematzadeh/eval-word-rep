#load the numba results

import pickle, sys, os, numpy as np
import scipy.stats
os.chdir('/home/stephan/notebooks/word-representations')
sys.path.append('/home/stephan/python/gibbs_lda')
import gibbs_lda

tsg_gs = pickle.load(open('tasa_skipgram_lda.pkl', 'rb'))

print dir(tsg_gs)

print tsg_gs.ZZ.shape
# crap, only 2 samples


tsg_doc_dist = tsg_gs.getTopicDistributionsForDocuments()

#topic distribution for documents code doesn't work with the current method

print(tsg_gs.ZZ[:,6000000]) #uh oh

Only a subset of topics?

topic_inventory = np.unique(tsg_gs.ZZ)
print len(topic_inventory)

# count the values in the matrix

scipy.stats import itemfreq
topic_freq = scipy.stats.itemfreq(tsg_gs.ZZ)
print topic_freq
#super concentrated on a small number of topics... smoothing problem?