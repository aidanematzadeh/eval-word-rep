# compare pickles for differences
import pickle
import os
import pandas as pd
import numpy as np
import pdb

comparison_path1 = "/shared_hd1/word-representations/results/dec-2017/cached/dec-2017"
comparison_path2 = "/shared_hd1/word-representations/results/dec-2017-2/cached/dec-2017-2"


def load_two_pickles(comparison_path1, comparison_path2, filename):
    path_1 = os.path.join(comparison_path1, filename)
    path_2 = os.path.join(comparison_path2, filename)
    pickle1 = pickle.load( open( path_1, "rb" ))
    pickle2 = pickle.load( open( path_2, "rb" ))
    return((pickle1, pickle2))



def check_allpairs():
    pickle1, pickle2 = load_two_pickles(comparison_path1, comparison_path2,  "allpairs.pkl")
    frame1 = pd.DataFrame()
    frame1['source1'] = [x[0] for x in pickle1]
    frame1['target1'] = [x[1] for x in pickle1]
    frame1 = frame1.sort_values(by=['source1','target1'])
    frame2 = pd.DataFrame()
    frame2['source2'] = [x[0] for x in pickle2]
    frame2['target2'] = [x[1] for x in pickle2]
    frame2 = frame2.sort_values(by=['source2','target2'])
    return(np.all(np.equal(frame1['source1'],frame2['source2'])) and (np.all(np.equal(frame1['target1'],frame2['target2']))))


def check_tuples():
	pickle1, pickle2 = load_two_pickles(comparison_path1, comparison_path2,  "tuples.pkl")

	# for each tuple, check if it in the other list
	presenceIn2 = []
	for tuple in pickle1:
	    presenceIn2.append(tuple in pickle2)

	presenceIn1 = []
	for tuple in pickle2:
	    presenceIn1.append(tuple in pickle1)

	return(np.all(presenceIn1 + presenceIn2))

def check_dataset(dataset, numToTest=20, checkOnly=True):
    pickle1, pickle2 = load_two_pickles(comparison_path1, comparison_path2,  dataset)
    same_count = []
    test_items = np.random.choice(list(pickle1.keys()), numToTest)
    for key in pickle1[test_item].keys():
        comparison = pickle1[test_item][key] == pickle2[test_item][key]
        if comparison:
            same_count.append(comparison)
        else:
            pdb.set_trace()
    success = np.all(same_count)
    if checkOnly:
        return(success)
    else:        
        pdb.set_trace()


def checkDatasets(datasets): # takes a long time because the intermediate data files are quite large
    dataset_success = []
    for dataset in datasets:
        dataset_success.append(check_dataset(dataset))
    return(pd.DataFrame({"dataset":np.array(datasets), 'success':np.array(dataset_success), "index":np.arange(len(datasets))}))


check_allpairs()
check_tuples()
checkDatasets(["pretrained-glove_cos.pickle",
"tasa-gibbslda_gibbslda.pickle",
"tasa-skipgram_cos.pickle",
"wiki-freq_freq.pickle",
"wiki-skipgram_cos.pickle",
"googlenews-w2v_cond.pickle",
"tasa-cbow_cond.pickle",
"tasa-glove_cond.pickle",
"wiki-glove_cond.pickle",
"googlenews-w2v_cos.pickle",
"tasa-cbow_cos.pickle",
"tasa-glove_cos.pickle",
"wiki-cbow_cond.pickle",
"wiki-glove_cos.pickle",
"pretrained-glove_cond.pickle",
"tasa-freq_freq.pickle",
"tasa-skipgram_cond.pickle",
"wiki-cbow_cos.pickle",
"wiki-skipgram_cond.pickle"])


check_dataset("tasa-gibbslda_gibbslda.pickle", checkOnly=False)
# should drop into pdb

# pickle1[test_item][key]
# 9.8347713351225602e-09
# pickle2[test_item][key]
# 9.8347713351225586e-09

#maybe for low probability items this could make a difference in ordering, esp. if this corresponds to small values like 1...5/n, where many words are likely to share a small n


