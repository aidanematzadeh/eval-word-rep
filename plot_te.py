# This script produces the traingle inequlity plots -- 

import pickle
import sys
import os
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

path = sys.argv[1]

te = {}

for filename in os.listdir(path):
    if not filename.endswith("te.pickle"):
        continue
    print(filename)
    name = filename[0:filename.find("_")]
    with open(path + "/" + filename, 'rb') as f:
        te[name] = pickle.load(f, encoding='latin1')

models_num = len(te)
thres_num = 4
f, ax = plt.subplots(thres_num, models_num)
#plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
f.subplots_adjust(hspace=0.3)
print(ax)
plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
plt.setp([a.get_xticklabels() for a in ax[-1,:]], visible=True)


namemap = {"Nelson Norms": "norms", "Topics": "tasa-gibbslda", "Frequency": "tasa-freq",
        "GloVe": "tasa-glove", "Word2Vec CBOW": "tasa-cbow", "Word2Vec Skip-gram": "tasa-skipgram"}
print(te.keys())
print(namemap.keys())
for mindex, name in enumerate(["Nelson Norms", "Topics", "Frequency", "GloVe", "Word2Vec CBOW", "Word2Vec Skip-gram"]):
    te_dist = te[namemap[name]]
    thresholds = [x for i,x in enumerate(sorted(te_dist.keys())) if i%2==0]
    xmax = numpy.max(te_dist[thresholds[0]])
    for index, thres in enumerate(thresholds):
        if name != "Frequency":
            ax[index, mindex].hist(te_dist[thres], label="%.5f" % (thres), normed=True)
            ax[index, mindex].xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        else:
            ax[index, mindex].xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax[index, mindex].hist(te_dist[thres], label="%d" % (thres), normed=True)
        ax[index, mindex].set_xlim(xmin=0)
        ax[index, mindex].set_xlim(xmax=xmax)
        if index == 0:
            ax[index, mindex].set_title(name, fontsize=7) #"i%s \n pairs %d" % (name, len(te_dist[thres])), fontsize=8)
        ax[index, mindex].legend(prop={'size':6})
        ax[index, mindex].set_xticks([xmax])
        ax[index, mindex].tick_params(axis='both', which='major', labelsize=5)

plt.show()
plt.savefig("%s_%d.png"% ("tasa", 1), dpi=600)

