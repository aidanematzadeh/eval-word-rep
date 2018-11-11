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
import sys

import process
import evaluate
import memory

def eval_model_worker(args):    

    model_ctrl, ctrl, norms, index = args
    print('Evaluating model:')
    print(model_ctrl)

    if model_ctrl['type'] == 'w2v':
        # word2vec inputs returns two outputs, cosine and conditional
        w2v_bin, w2v_cond = process.get_w2v(
            w2vcos_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_cos.pickle'),
            w2vcond_pickle = os.path.join(ctrl['cachePath'],model_ctrl['path']+'_cond.pickle'),
            norms = norms,
            w2v_path = os.path.join(ctrl['dataPath'], model_ctrl['path'],'model'),
            flavor = model_ctrl['flavor'],
            cond_eq = model_ctrl['condEq'],
            writePickle=True,
            regeneratePickle=model_ctrl['overwriteCache'] == 1)

        memory.memoryCheckpoint(index,'modelLoad')

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

        memory.memoryCheckpoint(index,'modelLoad')
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

        memory.memoryCheckpoint(index,'modelLoad')
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

        memory.memoryCheckpoint(index,'modelLoad')
        return([{'path':model_ctrl['path'],'data':freq}])
    else:
        raise NotImplementedError

def score_model_worker(args):
    stype, scores, allpairs, norms_assoc, norms, commonwords, gold_associates, asympairs, similarity_datasets = args #norms_asym

    # get the associations for each dataset
    print('Computing associations for ' + stype)
    rd = {}
    rd['scores'] = {'model_id':stype}        
    rd['associations'] = {}        

    if norms_assoc is None: #these are the norms themeselves; don't want to evaluate them
        for similarity_dataset in similarity_datasets:
            rd['scores'][similarity_dataset+'_correlation'] =  None
        
        model_associations = process.get_pair_scores(scores, allpairs['nelson_norms'])
        rd['associations']['nelson_norms'] = model_associations
        #!!! need to make sure norms are the first "model" that gets processed

    # bad interaction of the for loop with the norms as special case -- norms are missing key values
    else:        
       for similarity_dataset in similarity_datasets.keys():           
            model_associations = process.get_pair_scores(scores, allpairs[similarity_dataset])
            # allpairs[similarity_dataset] has a list of tuples
    
            if similarity_dataset == 'nelson_norms':
                rho = evaluate.rank_correlation(norms_assoc['associations']['nelson_norms'], model_associations)[0]
            else:
                # Here be dragons
                
                # using allpairs, extract a vector (dataest_associations) from similarity_datasets[similarity_dataset] to compare with model_associations
                pairs = [(pair[0],pair[1]) for pair in allpairs[similarity_dataset]]
                sim = similarity_datasets[similarity_dataset]
                dataset_associations = np.array([sim[pair[0]][pair[1]] for pair in pairs])
                
                rho = evaluate.rank_correlation(dataset_associations, model_associations)[0]
                #evaluate.rank_correlation(['associations'], model_associations)[0]                

            print('Associations for %s: %.2f' % (stype, rho))

            rd['scores'][similarity_dataset+'_correlation'] =  rho 
            rd['associations'][similarity_dataset] = model_associations

    #now set things with just the nelson_norms
    similarity_dataset = 'nelson_norms' 
    #note that these values are kept at the top level for scores
    print('Getting median ranks for ' + stype)
    
    if stype == 'norms': # !!! these are not reached in the initial evaluation of the norms
        #median rank is taken on just the items in target set
        scores_sorted = evaluate.sort_pairs(scores, allpairs[similarity_dataset])
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
    
    # commented out per Aida's recommendation on 9 November 2018 
    # else:
    #    for t in te_ratio:
    #        rd['scores']['te_rho_%.2f' % t] = evaluate.rank_correlation(norms_assoc['te'], te_ratio[t])[0]

    rd['scores']['te_dist'] = te_dist
    rd['scores']['sim_dist'] = sim_dist


    print('Getting ratio of asymmetries ' + stype)
    if stype.endswith("cos") or (stype in ('tasa-freq','wiki-freq')):
        # can't compute asym_rho for the frequency model (necessarily symmetrical)
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


    print('Create a single retrieval list')
    print('Getting norms...')
    norms = process.get_norms(ctrl['norms_pickle'], ctrl['norms_raw'], ('norms_pickle' in ctrl['regenerate']))
    #norms are cached at data/norms while the derived tuples are stored in cached/norms
    
    print('Augmenting norms with other similarity datasets')
    retrieval_list = norms.copy()
    
    similarity_datasets = {}
    for similarity_dataset in ctrl['similarity_datasets']:
        print('Loading '+similarity_dataset+'...')
        similarity_dataset_path = os.path.join(ctrl['similarityDatasetsPath'],similarity_dataset+'.txt')                
        sim_df = pd.read_table(similarity_dataset_path, header=None)
        if sim_df.shape[1] != 3:
            sim_df = pd.read_table(similarity_dataset_path, header=None, sep=' ')
        sim_df.columns = ['cue','target','value']

        similarity_datasets[similarity_dataset] = {}
        for record in sim_df.to_dict('records'):
            if record['cue'] in retrieval_list:
                if record['target'] in retrieval_list[record['cue']]:
                    pass # already retrieved
                else:
                    retrieval_list[record['cue']][record['target']] = record['value']                
            else:
                retrieval_list[record['cue']] = {}
                retrieval_list[record['cue']][record['target']] = record['value']

            if record['cue'] in similarity_datasets[similarity_dataset]:
                similarity_datasets[similarity_dataset][record['cue']][record['target']] = record['value']
            else:
                similarity_datasets[similarity_dataset][record['cue']] = {}
                similarity_datasets[similarity_dataset][record['cue']][record['target']] = record['value']
    
    similarity_datasets['nelson_norms'] = norms.copy()               

    cue_target_pairs = []
    for cue in retrieval_list.keys():
        for target in retrieval_list[cue]:
            cue_target_pairs.append({'cue':cue,'target':target})
    cue_target_df = pd.DataFrame(cue_target_pairs)        
    cue_target_df.to_csv('cue_target_pairs.csv')

    
    print('Retrieving similarities for %s models' % len(ctrl['models']))
    inputs = [(ctrl['models'][x], ctrl, retrieval_list, x) for x in range(len(ctrl['models']))] 

    # num_cores = multiprocessing.cpu_count() // 2
    num_cores = 12 #increasing this to any reasonable value causes a memory error on Chompsky

    # [ ] replace references to norms with "retrievalList" in process.py
    
    print('Multiprocessing with %s cores' % num_cores)
    #par_results = Parallel(n_jobs=num_cores)(delayed(eval_model_worker)(i) for i in inputs)
    par_results = [eval_model_worker(i) for i in inputs]

    memory.memoryCheckpoint(1, 'main')

    evallist = list(itertools.chain.from_iterable(par_results))

    print('Building a common test set for each evaluation...')
    allpairs = {}
    for similarity_dataset in similarity_datasets.keys():  
        print('Building common test set for '+similarity_dataset)
        allpairs[similarity_dataset] = process.get_allpairs_generalized(os.path.join(ctrl['cachePath'], similarity_dataset+'_allpairs.pkl'), similarity_datasets[similarity_dataset], [x['data'] for x in evallist], regeneratePickle=('allpairs' in ctrl['regenerate']))

    memory.memoryCheckpoint(2, 'main')


    asympairs = {}
    asympairs['nelson_norms'] = process.get_asym_pairs(norms, allpairs['nelson_norms'])

    memory.memoryCheckpoint(3, 'main')

    print("common pairs: %d, asym pairs: %d" % (len(allpairs['nelson_norms']), len(asympairs['nelson_norms'])))

    print('Reconciling vocabularies...')
    keys_per_model = [set(x['data'].keys()) for x in evallist]

    intersection_store = np.zeros([len(keys_per_model), len(keys_per_model)])

    for i in range(len(keys_per_model)):
        for j in range(i):
            intersection_store[i,j] = len(set.intersection(keys_per_model[i], keys_per_model[j]))

    memory.memoryCheckpoint(4, 'main')


    idf = pd.DataFrame(intersection_store, index = [x['path'] for x in evallist], columns= [x['path'] for x in evallist])       
    idf.to_csv(os.path.join(ctrl['resultsPath'],'key_overlap.csv'))

    commonwords =  set.intersection(*keys_per_model) 
    print("common cues", len(commonwords))
    tuples = process.get_tuples(ctrl['tuples_pickle'], norms, allpairs['nelson_norms'], regeneratePickle=('tuples' in ctrl['regenerate']))
    print("Number of Triangle Inequality tuples %d" % len(tuples))

    memory.memoryCheckpoint(5, 'main')

    gold_associates = evaluate.sort_pairs(norms, allpairs['nelson_norms'])

    print('Running tests')  
    #special case: generate the norms_assoc by scoring the norms. Don't need to include all similarity datasets here. 
    norms_assoc = score_model_worker(('norms', norms, allpairs, None, norms,
                                      commonwords, gold_associates, asympairs['nelson_norms'], similarity_datasets))
    score_inputs = [(x['path'], x['data'], allpairs, norms_assoc, norms,
                     commonwords, gold_associates, asympairs['nelson_norms'], similarity_datasets) for x in evallist]

    memory.memoryCheckpoint(6, 'main')

    print('Scoring models in parallel')
    #par_scores = Parallel(n_jobs=num_cores)(delayed(score_model_worker)(i) for i in score_inputs)
    par_scores = [score_model_worker(i) for i in score_inputs]

    memory.memoryCheckpoint(7, 'main')

    # plot the percentile rank
    te_data = [(x['scores']['model_id'], x['scores']['te_dist'], x['scores']['sim_dist']) for x in par_scores]
    evaluate.plot_percentile_rank(te_data, os.path.join(ctrl['resultsPath'], 'percentilerank.png'))

    memory.memoryCheckpoint(8, 'main')

    print('Saving results')
    # remove 'sim_dist', 'te_dist', and te_ro values for a managable output CSV
    
    # search for all column names
    score_df = pd.DataFrame([x['scores'] for x in par_scores])
    correlation_columns = [x for x in score_df.columns if '_correlation' in x]
    other_columns = ['model_id','asym_rho', 'median_found_rank_0', 'median_found_rank_1', 'median_found_rank_2', 'median_max_rank_0', 'median_max_rank_1', 'median_max_rank_2']    
    
    score_df[correlation_columns + other_columns].to_csv(os.path.join(os.path.join(ctrl['resultsPath'],'model_scores.csv')),index=False)
    
    memory.memoryCheckpoint(9, 'main')


