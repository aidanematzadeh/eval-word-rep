#!/usr/bin/env python3

import argparse
import collections
import os

import joblib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

def parse_args():
    parser = argparse.ArgumentParser(description='Produce the triangle inequality plots from the results.')
    parser.add_argument('results_dir', type=str,
                        help='the results directory')
    parser.add_argument('output_img', type=str,
                        help='the filename of the output plots (best as an SVG)')
    return parser.parse_args()

def main():
    args = parse_args()

    namemap = collections.OrderedDict([
        ('Nelson Norms', 'norms'),
        ('Topics', 'tasa-gibbslda'),
        ('Frequency', 'tasa-freq'),
        ('GloVe', 'tasa-glove'),
        ('Word2Vec CBOW', 'tasa-cbow'),
        ('Word2Vec Skip-gram', 'tasa-skipgram'),
    ])

    print('Namemap: {}'.format(list(namemap.values())))

    te = {}
    for filename in os.listdir(args.results_dir):
        if not filename.endswith('te.pickle'):
            continue

        name = filename.partition('_')[0]
        if name not in namemap.values():
            continue

        print(filename)
        te[name] = joblib.load(os.path.join(args.results_dir, filename))

    print('Te: {}'.format(list(te)))

    models_num = len(te)
    thres_num = 4
    f, ax = plt.subplots(nrows=thres_num, ncols=models_num)
    #plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
    f.subplots_adjust(hspace=0.3)
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    plt.setp([a.get_xticklabels() for a in ax[-1, :]], visible=True)

    for mindex, (name, te_name) in enumerate(namemap.items()):
        te_dist = te[te_name]
        thresholds = sorted(te_dist)[::2]
        xmax = np.max(te_dist[thresholds[0]])
        for index, thres in enumerate(thresholds):
            curr_ax = ax[index, mindex]
            if name != 'Frequency':
                curr_ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
                curr_ax.hist(te_dist[thres], label='%.5f' % (thres), normed=True)
            else:
                curr_ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                curr_ax.hist(te_dist[thres], label='%d' % (thres), normed=True)
            curr_ax.set_xlim(xmin=0)
            curr_ax.set_xlim(xmax=xmax)
            if index == 0:
                curr_ax.set_title(name, fontsize=7) #'i%s \n pairs %d' % (name, len(te_dist[thres])), fontsize=8)

            curr_ax.legend(prop={'size': 6})
            curr_ax.set_xticks([xmax])
            curr_ax.tick_params(axis='both', which='major', labelsize=5)

    plt.show()
    plt.savefig(args.output_img)

if __name__ == '__main__':
    main()
