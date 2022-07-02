import pickle
import random
import sys
import warnings
import copy
import numpy as np

from src.bnc import KDB
from src.experiments import experiment_only_Full

warnings.filterwarnings('ignore')

## SETTINGS
showTrace = False

num_full_labeled_insts = np.round((1+np.arange(10))*(100.0/3)).astype(int)
rel_prop_weak_labeled_insts = np.array([0.5, 1, 2, 5])
num_exp_reps_datasets = 30

num_cv_folds = 3
num_cv_reps = 5

nrls = np.arange(4,9)

path = "results/realdat_onlyfull/"


dtst = int(sys.argv[1])# <--------------------------- ARG 1 -------------------------
idx_nl = int(sys.argv[2])# <--------------------------- ARG 2 -------------------------

gen_mt = int(sys.argv[3])#1# <--------------------------- ARG 3 -------------------------

kDB_learn = int(sys.argv[4])#1# <--------------------------- ARG 4 -------------------------



type = "simple"; tp = "sg"
if gen_mt > 1:
    type = "complex"; tp = "cg"

dbfile = open('models_and_data/real_data_'+type+'.pkl', 'rb')
models = pickle.load(dbfile)
tranf_data = pickle.load(dbfile)
datasets = pickle.load(dbfile)
dbfile.close()

## RUN
f_real = open(path+'res_exp_realdat_fully_labeled_gen_'+tp+'_kdb_'+str(kDB_learn)+'_d_'+str(dtst)+'_nl_'+str(nrls[idx_nl])+'.resout', 'w', buffering=1)


rseed = (((5 + dtst)*7 + idx_nl)*11 + kDB_learn)*3
if type == "complex":
    rseed = rseed*17
np.random.seed(rseed)
random.seed(rseed)


gen_model = models[dtst][idx_nl]
cardinalities = gen_model.cardinalities
iClass = gen_model.class_ind

learn_model = KDB(iClass, cardinalities, K=kDB_learn)
if kDB_learn == gen_model.K:
    learn_model = copy.deepcopy(gen_model)
else:
    learn_model.random_generation()
learn_model.clear()


for d in np.arange(idx_nl*len(num_full_labeled_insts)*num_exp_reps_datasets,
                   (idx_nl+1)*len(num_full_labeled_insts)*num_exp_reps_datasets):
    print(dtst," ", nrls[idx_nl]," ", d)
    res_full = -np.ones(5)*np.inf

    if len(datasets[dtst][d]) > 0:
        idxs_sample = datasets[dtst][d][0]
        dataset = tranf_data[dtst][idx_nl][idxs_sample,:]
        # first_l_cs = create_genModel_gost_candidate_sets(datasets[d][0], gen_model)
        res_full = experiment_only_Full(learn_model, cardinalities, iClass, dataset,
                                            num_cv_folds, num_cv_reps, showTrace)
    else:
        print("no dataset")

    f_real.write(np.array2string(res_full, formatter={'float_kind':lambda x: "%.4e" % x}) + '\n')

f_real.close()

