# word_representations
This repository contains code to evaluate machine-generated word representations against human behavior, specifically the Nelson assocaition norms. This repository does not contain code to generate such models (see instead https://github.com/smeylan/batch_w2v), or to choose among such models the one that is the best fit to human data.

We intend for this code to be run under Python3, with or without a virtual environment. To install the requirements, `pip install -r requirements.txt`

Rather than using command line arguments, the main analysis script `eval_parallel.py` takes a .json control file as input. This control file specifies the paths to the models that are to be evaluated, the appropriate directory to place the results, and whether intermediate files (either for the models or for the sets of similarity judgments being compared) are to be cached. Caching will use a relatively large amount of disk space (e.g., 11 gb for 10 models). 
