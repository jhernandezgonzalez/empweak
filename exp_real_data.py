import pickle
import random
import numpy as np
import copy
import sys
import os
import warnings

from src.bnc import KDB
from src.experiments import experiment_only_Weak

warnings.filterwarnings('ignore')

## SETTINGS

showTrace = False

num_full_labeled_insts = np.round((1 + np.arange(10)) * (100.0 / 3)).astype(int)
rel_prop_weak_labeled_insts = np.array([0.5, 1, 2, 5])
num_exp_reps_datasets = 30
nrls = np.arange(4,9)

num_cv_folds = 3
num_cv_reps = 5

path = "results/realdat_weak/"


dtst = int(sys.argv[1])# <--------------------------- ARG 1 -------------------------
idx_nl = int(sys.argv[2])# <--------------------------- ARG 2 -------------------------

num_dataset = int(sys.argv[3])# <--------------------------- ARG 3 -------------------------

gen_mt = int(sys.argv[4])#1# <--------------------------- ARG 4 -------------------------

kDB_learn = int(sys.argv[5])#1# <--------------------------- ARG 5 -------------------------

full_set_size = int(sys.argv[6]) # <--------------------------- ARG 6 -------------------------

type = "simple"; tp = "sg"
if gen_mt > 1:
    type = "complex"; tp = "cg"


filename = 'res_exp_realdat_weak_gen_'+tp+'_kdb_'+str(kDB_learn)+'_d_'+str(dtst)+\
           '_nl_'+str(nrls[idx_nl])+'_rep_'+str(num_dataset)+'_fss_'+str(full_set_size)+'.csv'

# We don't want to run it several times.
if os.path.isfile(path+filename):
    sys.exit(0)

dbfile = open('models_and_data/real_data_'+type+'.pkl', 'rb')
orig_models = pickle.load(dbfile)
orig_tranf_data = pickle.load(dbfile)
orig_datasets = pickle.load(dbfile)
orig_tranf_csets = pickle.load(dbfile)
dbfile.close()


## RUN
rseed = (((((5 + dtst)*7 + num_dataset)*11 + full_set_size)*13 + kDB_learn)*17+idx_nl*3)
if type == "complex":
    rseed = rseed*21
np.random.seed(rseed)
random.seed(rseed)

gen_model = orig_models[dtst][idx_nl]
cardinalities = gen_model.cardinalities
iClass = gen_model.class_ind

learn_model = KDB(iClass, cardinalities, K=kDB_learn)
if kDB_learn == gen_model.K:
    learn_model = copy.deepcopy(gen_model)
else:
    learn_model.random_generation() # just to obtain the default kDB structure
learn_model.clear()

gen_res_weak = []

d = (idx_nl*num_exp_reps_datasets+num_dataset)*len(num_full_labeled_insts)+full_set_size

if len(orig_datasets[dtst][d]) > 0:
    idxs_sample = orig_datasets[dtst][d][0]

    for ind_wlp in np.arange(len(rel_prop_weak_labeled_insts)):
        print(ind_wlp)
        if len(orig_datasets[dtst][d][ind_wlp + 1]) > 0:
            idxs_sample = np.concatenate((idxs_sample, orig_datasets[dtst][d][ind_wlp + 1]))
            idxs_sample.sort()
            dataset = orig_tranf_data[dtst][idx_nl][idxs_sample, :]
            cand_sets = [ orig_tranf_csets[dtst][idx_nl][i] for i in idxs_sample ]

            res_weak = experiment_only_Weak(learn_model, cardinalities, iClass, dataset, cand_sets,
                                            num_cv_folds, num_cv_reps, showTrace)

            gen_res_weak.append( res_weak )
        else:
            gen_res_weak.append(-np.ones(5)*np.inf)
            print("no weak dataset")
else:
    for i in np.arange(4):
        gen_res_weak.append(-np.ones(5) * np.inf)
    print("no full dataset")

np.savetxt(path+filename,
           np.array(gen_res_weak), delimiter=",", fmt='%.4e')
