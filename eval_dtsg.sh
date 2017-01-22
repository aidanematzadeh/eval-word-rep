#!/bin/bash

# Where to look for pickle files
PKL=$1

TSGPOS=$2
TSGNEG=$3

NORMS="data/nelson_norms/"
CBOWBIN="data/GoogleNews-vectors-negative300.bin"

W2ID="data/wikipedia_sw_norms_100k/5w_word2id"
CNTS="data/wikipedia_sw_norms_100k/5w_positive_counts"
IDS="data/wikipedia_sw_norms_100k/5w_positive_ids"

GLOVE="test_results/glove/glove.6B.300d.txt"
SG="test_results/sg/size-200_window-5_mc-0_workers-12_sg-0_neg-0_hs-1"

echo $NORMS
echo $PKL/norms.pickle


python3.5 eval_dtsg.py \
--norms_pickle $PKL/norms.pickle \
--norms_dirpath $NORMS \
--cbowcos_pickle $PKL/cbowcos.pickle \
--cbowcond_pickle $PKL/cbowcond.pickle \
--cbow_binarypath $CBOWBIN \
--cbowcond_eq eq1 \
--sgcos_pickle $PKL/sgcos.pickle \
--sgcond_pickle $PKL/sgcond.pickle \
--sg_path $SG \
--sgcond_eq eq1 \
--tsg_vocabpath $W2ID \
--tsg_countspath $CNTS \
--tsg_idspath $IDS \
--tsgfreq_pickle $PKL/tsgfreq.pickle \
--tsgpos_pickle $PKL/tsgpos.pickle \
--tsgpos_gammapath $TSGPOS/gamma7 \
--tsgpos_lambdapath $TSGPOS/lambda7 \
--tsgneg_pickle $PKL/tsgneg.pickle \
--tsgnegnorm_pickle $PKL/tsgnegnorm.pickle \
--tsgneg_gammapath $TSGNEG/gamma4916 \
--tsgneg_lambdapath $TSGNEG/lambda100154 \
--tsgneg_mupath $TSGNEG/mu100154 \
--glovecos_pickle $PKL/glovecos.pickle \
--glovecond_pickle $PKL/glovecond.pickle \
--glove_path $GLOVE \
--allpairs_pickle $PKL/allpairs_sg.pickle \
--outdir $PKL/ \

# vim /home/amint/.local/lib/python3.5/site-packages/gensim/utils.py
# Aida: I editted the script so you can just run this ./eval_dtsg.sh test_results/jan6pkl test_results/jan6pos/ test_results/jan6neg/
#tr '\n' ' ' < eval_dtsg.sh > eval_dtsg_run.sh && chmod +x eval_dtsg_run.sh && ./eval_dtsg_run.sh
