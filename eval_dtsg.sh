python3 eval_dtsg.py
--norms_pickle test_results/dec14pos/norms.pickle
--norms_dirpath data/nelson_norms/
--cbowcos_pickle test_results/dec14pos/cbowcos.pickle
--cbowcond_pickle test_results/dec14pos/cbowcond.pickle
--cbow_binarypath GoogleNews-vectors-negative300.bin
--sgcos_pickle test_results/dec14pos/sgcos.pickle
--sgcond_pickle test_results/dec14pos/sgcond.pickle
--sg_path w2v/size-400_window-5_mc-0_workers-12_sg-0_neg-15_hs-0
--lda_pickle test_results/dec14pos/lda.pickle
--ldavocab_path data/wikipedia_sw_norms_100k/5w_word2id
--ldagamma_path dec14_pos/gamma4916
--ldalambda_path dec14_pos/lambda100154
--ldamu_path none
--allpairs_pickle test_results/dec14pos/allpairs_sg.pickle
--outdir test_results/dec14pos/
#tr '\n' ' ' < eval_dtsg.sh > eval_dtsg_run.sh && chmod +x eval_dtsg_run.sh && ./eval_dtsg_run.sh
