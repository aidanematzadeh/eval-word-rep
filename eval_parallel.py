#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import pandas as pd
import itertools
import json
import multiprocessing
import joblib
from joblib import Parallel, delayed

import process
import evaluate

def eval_model_worker(args):
    model_ctrl, ctrl, norms = args

    if model_ctrl['type'] == 'w2v':
        # word2vec inputs returns two outputs, cosine and conditional
        w2v_bin, w2v_cond = process.get_w2v(
            w2vcos_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_cos.pickle'),
            w2vcond_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_cond.pickle'),
            norms = norms,
            w2v_path = os.path.join(ctrl['dataPath'], model_ctrl['path'],'model'),
            binary_flag = model_ctrl['bin'] == 1,
            cond_eq = model_ctrl['condEq'],
            writePickle=True,
            regeneratePickle=model_ctrl['overwriteCache'] == 1)

        return([{'path':model_ctrl['path']+'_cos', 'data': w2v_bin}, { 'path': model_ctrl['path']+'_cond', 'data':w2v_cond}])

    elif model_ctrl['type'] == 'glove':
        glove_cos, glove_cond =  process.get_glove(
            glovecos_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_cos.pickle'),
            glovecond_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_cond.pickle'),
            glove_path = os.path.join(ctrl['dataPath'], model_ctrl['path'],'vectors.txt'),
            norms = norms,
            cond_eq = model_ctrl['condEq'],
            writePickle=True,
            regeneratePickle=model_ctrl['overwriteCache'] == 1)
        return([{'path':model_ctrl['path']+'_cos', 'data':glove_cos}, {'path':model_ctrl['path']+'_cond', 'data':glove_cond}])

    elif model_ctrl['type']    == 'gibbslda':
        #
        gibbslda = process.get_gibbslda_avg(
            gibbslda_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_gibbslda.pickle'),
            beta = 0.01,
            norms = norms,
            vocab_path = os.path.join(ctrl['dataPath'],model_ctrl['path'],model_ctrl['vocab_path']),
            lambda_path  = os.path.join(ctrl['dataPath'],model_ctrl['path'], model_ctrl['lambda_path']),
            writePickle=True,
            regeneratePickle=model_ctrl['overwriteCache'] == 1)
        return([{'path':model_ctrl['path'], 'data':gibbslda}])

    elif model_ctrl['type'] == 'freq':
        #{"path": "tasa-freq", "type": "freq", "overwriteCache": 0, "counts_path":"5w_positive_counts", "vocab_path":"5w_word2id", "ids_path":"5w_positive_ids"},
        freq = process.get_tsgfreq(
            tsgfreq_pickle = os.path.join(ctrl['cachePath'], model_ctrl['path']+'_freq.pickle'),
        norms = norms,
        vocab_path = os.path.join(ctrl['dataPath'], model_ctrl['path'], model_ctrl['vocab_path']),
        counts_path = os.path.join(ctrl['dataPath'], model_ctrl['path'], model_ctrl['counts_path']),
        ids_path = os.path.join(ctrl['dataPath'],model_ctrl['path'], model_ctrl['ids_path']),
        writePickle=True,
        regeneratePickle=model_ctrl['overwriteCache'] == 1)
        return([{'path':model_ctrl['path'],'data':freq}])
    else:
        raise NotImplementedError

def score_model_worker(args):
    stype, scores, allpairs, norms_assoc, norms, commonwords, gold_associates, asympairs = args #norms_asym

    # get the associations
    print('Computing associations for ' + stype)
    model_associations = process.get_pair_scores(scores, allpairs)
    if norms_assoc is None: #these are the norms; can't compare to the norms
        rho = 1
    else:
        rho = evaluate.rank_correlation(norms_assoc['associations'], model_associations)[0]
    print('Associations for %s: %.2f' % (stype, rho))

    rd = {}
    rd['scores'] = {'model_id':stype, 'correlation': rho, }
    rd['associations'] = model_associations

    print('Getting median ranks for ' + stype)
    if stype == 'norms':
        #median rank is taken on just the items in target set
        scores_sorted = evaluate.sort_pairs(scores, allpairs)
    else:
        # longer median rank computation -- all norms and cues
        scores_sorted = evaluate.sort_all(scores, norms, commonwords)
    ranks, maxranks = evaluate.median_rank(gold_associates, scores_sorted, 3)
    for rank in ranks:
        rd['scores']['median_found_rank_%s' % rank] = np.median(ranks[rank])
        rd['scores']['median_max_rank_%s' % rank] = np.median(maxranks[rank])

    print('Getting triangle inequality results for '+ stype)
    te_dist, sim_dist, te_ratio = evaluate.traingle_inequality_threshold(stype, tuples, scores) #, commonwords, threshs)
    with open(os.path.join(ctrl['resultsPath'], stype + "_te.pickle"), 'wb') as output:
        joblib.dump(te_dist, output)

    evaluate.plot_traingle_inequality(te_dist, sim_dist,
                                      os.path.join(ctrl['resultsPath'], stype + "_te."))
    if norms_assoc is None: #these are the norms; can't compare to the norms
        rd['te'] = te_ratio
        rd['scores']['te_rho'] = 1
    else:
        for t in te_ratio:
            rd['scores']['te_rho_%.2f' % t] = evaluate.rank_correlation(norms_assoc['te'], te_ratio[t])[0]

    rd['scores']['te_dist'] = te_dist
    rd['scores']['sim_dist'] = sim_dist


    print('Getting ratio of asymmetries ' + stype)
    if stype.endswith("cos"):
        rd['scores']['asym_rho'] = None
    else:
        asyms = {}
        asyms["ratio"], asyms["difference"] = evaluate.asymmetry(scores, asympairs)
        if norms_assoc is None: #these are the norms; can't compare to the norms
            rd['asyms'] = asyms["ratio"]
            rd['scores']['asym_rho'] = 1
        else:
            rd['scores']['asym_rho'] = evaluate.rank_correlation(norms_assoc['asyms'], asyms['ratio'])[0]

    return(rd)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ctrl", type=str, help="name of .ctrl json")
    args = argparser.parse_args()

    print('Loading control .json')
    with open(args.ctrl) as ctrl_json:
        ctrl = json.load(ctrl_json)
    ctrl['norms_pickle'] =  os.path.join(ctrl['normsPath'],'norms.pkl')
    ctrl['norms_raw'] =  os.path.join(ctrl['normsPath'], 'raw')
    ctrl['cachePath'] = os.path.join(ctrl['cacheDir'], ctrl['runname'])
    ctrl['resultsPath'] = os.path.join(ctrl['resultsDir'], ctrl['runname'])
    ctrl['allpairs_pickle'] = os.path.join(ctrl['cachePath'], 'allpairs.pkl')
    ctrl['tuples_pickle'] = os.path.join(ctrl['cachePath'], 'tuples.pkl')
    #!!! format checks on the json
    #!!! confirm all models are of a supported type
    # if not args.model_type in ('cbow','sg','tsg','glove'):
            # raise ValueError('Specify one of the following model types: cbow, sg, tsg, glove')
    # confirm that  all of the expected model files are there

    print('Setting up paths...')
    for path in (ctrl['dataPath'], ctrl['normsPath'], ctrl['norms_raw']):
        if not os.path.exists(path):
            raise ValueError('Path '+ path +' must exist')

    for path in (ctrl['cachePath'], ctrl['resultsPath']):
        if not os.path.exists(path):
            os.makedirs(path)

    print('Getting norms...')
    norms = process.get_norms(ctrl['norms_pickle'], ctrl['norms_raw'], ('norms_pickle' in ctrl['regenerate']))
    #norms are cached at data/norms while the derived tuples are stored in cached/norms

    print('Retrieving similarities for %s models' % len(ctrl['models']))
    inputs = [(x, ctrl, norms) for x in ctrl['models']]

    # num_cores = multiprocessing.cpu_count() // 2
    num_cores = 1

    print('Multiprocessing with %s cores' % num_cores)
    par_results = Parallel(n_jobs=num_cores)(delayed(eval_model_worker)(i) for i in inputs)

    evallist = list(itertools.chain.from_iterable(par_results))

    print('Building a common test set...')
    allpairs = process.get_allpairs_generalized(ctrl['allpairs_pickle'], norms, [x['data'] for x in evallist], regeneratePickle=('allpairs' in ctrl['regenerate']))
    asympairs = process.get_asym_pairs(norms, allpairs)
    print("common pairs: %d, asym pairs: %d" % (len(allpairs), len(asympairs)))

    print('Reconciling vocabularies...')
    keys_per_model = [set(x['data'].keys()) for x in evallist]

    intersection_store = np.zeros([len(keys_per_model), len(keys_per_model)])

    for i in range(len(keys_per_model)):
        for j in range(i):
            intersection_store[i,j] = len(set.intersection(keys_per_model[i], keys_per_model[j]))


    idf = pd.DataFrame(intersection_store, index = [x['path'] for x in evallist], columns= [x['path'] for x in evallist])       
    idf.to_csv(os.path.join(ctrl['resultsPath'],'key_overlap.csv'))



    commonwords =  set.intersection(*keys_per_model) 
    print("common cues", len(commonwords))
    tuples = process.get_tuples(ctrl['tuples_pickle'], norms, allpairs, regeneratePickle=('tuples' in ctrl['regenerate']))
    print("Number of Triangle Inequality tuples %d" % len(tuples))

    gold_associates = evaluate.sort_pairs(norms, allpairs)

    print('Running tests')
    norms_assoc = score_model_worker(('norms', norms, allpairs, None, norms,
                                      commonwords, gold_associates, asympairs))
    score_inputs = [(x['path'], x['data'], allpairs, norms_assoc, norms,
                     commonwords, gold_associates, asympairs) for x in evallist]

    print('Scoring models in parallel')
    par_scores = Parallel(n_jobs=num_cores)(delayed(score_model_worker)(i) for i in score_inputs)

    # plot the percentile rank
    te_data = [(x['scores']['model_id'], x['scores']['te_dist'], x['scores']['sim_dist']) for x in par_scores]
    evaluate.plot_percentile_rank(te_data, os.path.join(ctrl['resultsPath'], 'percentilerank.png'))



    print('Saving results')
    score_df = pd.DataFrame([x['scores'] for x in par_scores])
    score_df.to_csv(os.path.join(os.path.join(ctrl['resultsPath'],'model_scores.csv')),index=False)
