import pickle
import random
import numpy as np
import copy
import warnings

from src.bnc import KDB
from src.experiments import experiment_only_Full

warnings.filterwarnings('ignore')

## SETTINGS

path = "results/figB/"

dbfile = open('models_and_data/figB_complexgen.pkl', 'rb')

models = pickle.load(dbfile)
conf_mtrcs = pickle.load(dbfile)
datasets = pickle.load(dbfile)

dbfile.close()

num_cv_folds = 3
num_cv_reps = 5
kDB_learn = 1

showTrace = False


## RUN

f_real = open(path+'res_exp_figB_cg_fully_labeled.resout', 'w', buffering=1)

num_full_ds = 10
num_models = len(models)
num_datasets = int(len(datasets) / num_models / num_full_ds)

rseed = ((5 + num_models)*7 + num_datasets)*11 + kDB_learn
np.random.seed(rseed)
random.seed(rseed)

for m in np.arange(num_models):
    gen_model = models[m]
    cardinalities = gen_model.cardinalities
    iClass = gen_model.class_ind

    learn_model = KDB(iClass, cardinalities, K=kDB_learn)
    learn_model.random_generation()
    learn_model.clear()

    for d in np.arange(m*num_datasets*num_full_ds,(m+1)*num_datasets*num_full_ds):
        print(m," ", d)
        dataset = copy.deepcopy(datasets[d][0])
        #first_l_cs = create_genModel_gost_candidate_sets(datasets[d][0], gen_model)
        res_full = experiment_only_Full(learn_model, cardinalities, iClass, dataset,
                                            num_cv_folds, num_cv_reps, showTrace)

        f_real.write(np.array2string(res_full, formatter={'float_kind':lambda x: "%.4e" % x}) + '\n')

f_real.close()

