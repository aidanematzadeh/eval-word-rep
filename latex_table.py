#!/usr/bin/env python3

import argparse
import os
import pandas as pd

from collections import OrderedDict

MISSING = '[*]'

def indices(d):
    return pd.Index(list(d.values())).unique()

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
    cols = ['source', 'model', 'simtype', 'correlation']
    new_df = (df[cols]
              .replace({
                  'source': FULL_NAME_OF_SOURCE,
                  'model': FULL_NAME_OF_MODEL,
                  'simtype': NAME_OF_SIMTYPE,
              })
              .reset_index()

              # Rearrange the table by the source and model.
              .pivot_table(index='source',
                           columns=['model', 'simtype'],
                           values='correlation')
              .reindex(FULL_SOURCE_NAMES, axis='index')
              .reindex(FULL_MODEL_NAMES, axis='columns', level=0)
              .reindex(SIMTYPE_NAMES, axis='columns', level=1)
              .round(2)
              .fillna(MISSING))

    new_df.index.name = None
    new_df.columns.names = [None, None]
    return new_df

def table2(df):
    # Remove unused rows and columns.
    median_cols = ['median_found_rank_{}'.format(i) for i in range(3)]
    cols = ['source', 'model'] + median_cols
    new_df = (df[cols][df['simtype'] != 'cos']
              .replace({
                  'source': FULL_NAME_OF_SOURCE,
                  'model': FULL_NAME_OF_MODEL,
              })
              .reset_index())

    # Merge the median columns together.
    new_df['median'] = (pd.Series(new_df[median_cols]
                                  .astype(str)
                                  .values.tolist())
                        .str
                        .join('/'))
    new_df = (new_df
              .drop(median_cols, axis='columns')

              # Rearrange the table by the source and model.
              .pivot(index='source', columns='model', values='median')
              .reindex(FULL_SOURCE_NAMES, axis='index')
              .reindex(FULL_MODEL_NAMES, axis='columns')
              .fillna(MISSING))

    new_df.index.name = None
    new_df.columns.name = None
    return new_df

def table3(df):
    # Remove unused rows and columns.
    cols = ['source', 'model', 'asym_rho']
    new_df = (df[cols][df['simtype'] != 'cos']
              .replace({
                  'source': SHORT_NAME_OF_SOURCE,
                  'model': FULL_NAME_OF_MODEL,
              })
              .reset_index()

              # Rearrange the table by the source and model.
              .pivot(index='source', columns='model', values='asym_rho')
              .reindex(SHORT_SOURCE_NAMES, axis='index')
              .reindex(FULL_MODEL_NAMES, axis='columns')
              .dropna(how='all', axis='columns')
              .round(2)
              .abs()
              .fillna(MISSING))

    new_df.index.name = None
    new_df.columns.name = None
    return new_df

def read_csv(fname):
    df = pd.read_csv(fname)

    # Break "wiki-skipgram_cos" into "wiki", "skipgram", "cos".
    def split_model_id(model_id):
        source, model_simtype = model_id.split('-', 1)
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
    os.makedirs(args.out_dir, exist_ok=True)
    for i, table_fn in enumerate([table1, table2, table3]):
        table_df = table_fn(df)
        latex_table = to_latex(table_df)
        fname = os.path.join(args.out_dir, 'table{}.tex'.format(i + 1))
        with open(fname, 'w') as f:
            f.write(latex_table)

if __name__ == '__main__':
    main()
