#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np

from collections import OrderedDict

MISSING = '[*]'

def indices(d):
    return pd.Index(list(d.values())).unique()


#JP: So when my files say cbow they always mean gnews. When they say glove they mean 840b. When they say skipgram it's Wikipedia. So skipgram with graph in the filename is faruqui tuning on top of a Wikipedia+skipgram model

FIX_MODEL_ID = {
	"pretrained-glove_cos" : "googlenews?-glove_cos", # what is the "pretrained" dataset? GoogleNews?
	"pretrained-glove_cond" : "googlenews?-glove_cond",
	"cbow_raw_cos": "googlenews-cbow.raw_cos",
	"cbow_raw_cond" : "googlenews-cbow.raw_cond", 
	"glove_raw_cos": "googlenews-glove.raw_cos", 
	"glove_raw_cond": "googlenews-glove.raw_cond", 
	"skipgram_raw_cos": "wiki-skipgram.raw_cos",
	"skipgram_raw_cond": "wiki-skipgram.raw_cond",
	"tuned_cbow_2e60_cos": "googlenews-tuned.cbow.2e60_cos",
	"tuned_cbow_2e60_cond": "googlenews-tuned.cbow.2e60_cond",
	"tuned_cbow_e19_cos": "googlenews-tuned.cbow.e19_cos",
	"tuned_cbow_e19_cond": "googlenews-tuned.cbow.e19_cond",
	"tuned_cbow_graph_cos": "googlenews-tuned.cbow.graph_cos",
	"tuned_cbow_graph_cond": "googlenews-tuned.cbow.graph_cond",
	"tuned_cbow_e47_cos": "googlenews-tuned.cbow.e47_cos",
	"tuned_cbow_e47_cond": "googlenews-tuned.cbow.e47_cond",
	"tuned_glove_graph_cos": "googlenews-tuned.glove.graph_cos",
	"tuned_glove_graph_cond": "googlenews-tuned.glove.graph_cond",
	"tuned_skipgram_e28_cos": "wiki-tuned.skipgram.e28_cos",
	"tuned_skipgram_e28_cond": "wiki-tuned.skipgram.e28_cond",
	"tuned_skipgram_graph_cos": "wiki-tuned.skipgram.graph_cos",
	"tuned_skipgram_graph_cond": "wiki-tuned.skipgram.graph_cond",
	"cbowe19_alpha0_cos": "googlenews-cbowe19.alpha0_cos",
	"cbowe19_alpha0_cond": "googlenews-cbowe19.alpha0_cond",
	"cbowe26_alpha0.25_cos": "googlenews-cbowe26.alpha0.25_cos" ,
	"cbowe26_alpha0.25_cond": "googlenews-cbowe26.alpha0.25_cond" ,
	"cbowe79_alpha0.75_cos": "googlenews-cbowe79.alpha0.75_cos",
	"cbowe79_alpha0.75_cond": "googlenews-cbowe79.alpha0.75_cond",
	"cbowe39_alpha0.5_cos": "googlenews-cbowe39.alpha0.5_cos",
	"cbowe39_alpha0.5_cond": "googlenews-cbowe39.alpha0.5_cond"
}


FULL_NAME_OF_SOURCE = OrderedDict([
    ('tasa', 'Small (TASA)'),
    ('wiki', 'Medium (Wikipedia)'),
    ('googlenews', 'Largest available'),
    ('pretrained', 'Largest available'),
])

FULL_SOURCE_NAMES = indices(FULL_NAME_OF_SOURCE)

SHORT_NAME_OF_SOURCE = OrderedDict([
    ('tasa', 'Small'),
    ('wiki', 'Medium'),
    ('googlenews?', 'Largest avail.'),
    ('googlenews', 'Largest avail.'),
    ('pretrained', 'Largest avail.'),    
])

SHORT_SOURCE_NAMES = indices(SHORT_NAME_OF_SOURCE)

NAME_OF_SIMTYPE = OrderedDict([
    ('cos', 'cos.'),
    ('cond', 'cond. pr.'),
    (None, '-'),
])

SIMTYPE_NAMES = indices(NAME_OF_SIMTYPE)

FULL_NAME_OF_MODEL = OrderedDict([
    ('cbow', 'Word2Vec CBOW'),
    ('w2v', 'Word2Vec CBOW'),
    ('skipgram', 'Word2Vec skip-gram'),
    ('glove', 'GloVe'),
    ('gibbslda', 'LDA'),
    ('freq', 'Co-occurrence'),
])

FULL_MODEL_NAMES = indices(FULL_NAME_OF_MODEL)

def table1(df):
    # Remove unused columns.

    '''
    scores = ['EN-MC-30_correlation', 'EN-MEN-TR-3k_correlation',	
	'EN-MTurk-287_correlation',	'EN-MTurk-771_correlation',
	'EN-RG-65_correlation',	'EN-RW-STANFORD_correlation',
	'EN-WS-353-ALL_correlation', 'EN-WS-353-REL_correlation',
	'EN-WS-353-SIM_correlation', 'EN-YP-130_correlation',
	'nelson_norms_correlation']
	'''
    scores = ['nelson_norms_correlation']	

    cols = ['source', 'model', 'simtype'] + scores

    # get it to longform
    new_df = (df[cols]
              .replace({
                  'source': FULL_NAME_OF_SOURCE,
                  'model': FULL_NAME_OF_MODEL,
                  'simtype': NAME_OF_SIMTYPE,
              }))

    # take the subset of models that we want for table 1
    t1_models = ['Word2Vec CBOW', 'Word2Vec skip-gram','GloVe', 'LDA', 'Co-occurrence']
    new_df = new_df.loc[new_df.model.isin(t1_models)]


    new_df = new_df.reset_index().pivot_table(index='source',
      			columns=['model', 'simtype'],
      			values=scores).reindex(FULL_SOURCE_NAMES, axis='index').round(2).fillna(MISSING)

    return new_df

def roundIfPossible(x, stringify=False):		 
	if (x.astype(int) == x).all():
		rv  = x.astype(int)
	else:
		rv = np.round(x, decimals=1)
	if stringify:
		return(str(rv))
	else:
		return(rv)

def table2(df):
    # Remove unused rows and columns.
    median_cols = ['median_found_rank_{}'.format(i) for i in range(3)]

    cols = ['source', 'model','simtype'] + median_cols
    new_df = (df.replace({
                  'source': FULL_NAME_OF_SOURCE,
                  'model': FULL_NAME_OF_MODEL,
                  'simtype': NAME_OF_SIMTYPE
              })
              .reset_index())

    # Merge the median columns together.
    new_df['median'] = np.apply_along_axis(lambda y: "/".join([roundIfPossible(x, stringify=True) for x in y]), 1, new_df[median_cols])
    
    new_df = (new_df
              .drop(median_cols, axis='columns'))

    t2_models = ['Word2Vec CBOW', 'Word2Vec skip-gram','GloVe', 'LDA', 'Co-occurrence']
    new_df = new_df.loc[new_df.model.isin(t2_models)]


    table_form = new_df.reset_index().pivot_table(index='source', columns=['model', 'simtype'], values=['median'], aggfunc = lambda x: x).fillna(MISSING).reindex(FULL_SOURCE_NAMES, axis='index')
    
    return table_form

def table3(df):
    # Remove unused rows and columns.
    cols = ['source', 'model', 'asym_rho', 'simtype']
    
    new_df = (df[cols]
              .replace({
                  'source': SHORT_NAME_OF_SOURCE,
                  'model': FULL_NAME_OF_MODEL,
                  'simtype': NAME_OF_SIMTYPE
              })
              .reset_index())
              # Rearrange the table by the source and model.
    del new_df['index']

    new_df['asym_rho'] = new_df['asym_rho'].abs().round(decimals=2)

    t3_models = ['Word2Vec CBOW', 'Word2Vec skip-gram','GloVe', 'LDA', 'Co-occurrence']
    new_df = new_df.loc[new_df.model.isin(t3_models)]


    import pdb
    pdb.set_trace()

    new_df = new_df.pivot_table(index='source', columns=['model', 'simtype'], values=['asym_rho'], aggfunc = lambda x: x ).fillna(MISSING)

    
    return new_df

def read_csv(fname):
	df = pd.read_csv(fname)
	df = df.replace({'model_id': FIX_MODEL_ID})

	# Break "wiki-skipgram_cos" into "wiki", "skipgram", "cos".
	def split_model_id(model_id):
		if len(model_id.split('-', 1)) == 2:
			source, model_simtype = model_id.split('-', 1)        
		else:
			# if the model name does not follow the source-model_measure convention       
			source = 'wiki' #!!! confirm that this is true
			model_simtype = model_id

		model_simtype = model_simtype.split('_', 1)
		if len(model_simtype) == 1:
			model_simtype.append(None)
		model, simtype = model_simtype		
		return source, model, simtype


	df['source'], df['model'], df['simtype'] = (df['model_id']
                                                .map(split_model_id)
                                                .str)


	del df['model_id']
	return df

def to_latex(df):
    # Center all columns.
    cols_count = df.shape[1] + 1
    return df.to_latex(column_format='c' * cols_count,
                       multicolumn=True,
                       multicolumn_format='c',
                       multirow=True)

def main():
    parser = argparse.ArgumentParser(description='Convert a CSV file to a LaTeX table.')
    parser.add_argument('--in', type=str,
                        required=True,
                        dest='in_fname',
                        help='the input CSV file name')
    parser.add_argument('--out', type=str,
                        required=True,
                        dest='out_dir',
                        help='the output dir for the LaTeX files')
    args = parser.parse_args()

    df = read_csv(args.in_fname)
    if not os.path.exists(args.out_dir):
    	os.makedirs(args.out_dir)	
    for i, table_fn in enumerate([table1, table2, table3]):
        table_df = table_fn(df)
        latex_table = to_latex(table_df)
        fname = os.path.join(args.out_dir, 'table{}.tex'.format(i + 1))
        with open(fname, 'w') as f:
            f.write(latex_table)

if __name__ == '__main__':
    main()
