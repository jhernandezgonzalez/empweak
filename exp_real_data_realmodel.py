import pickle
import random
import warnings
import sys
import os
import numpy as np

from src.experiments import only_test
from src.bnc import KDB

warnings.filterwarnings('ignore')

## SETTINGS
showTrace = False

num_full_labeled_insts = np.round((1+np.arange(10))*(100.0/3)).astype(int)
rel_prop_weak_labeled_insts = np.array([0.5, 1, 2, 5])
num_exp_reps_datasets = 30

num_cv_folds = 3
num_cv_reps = 5

nrls = np.arange(4,9)

path = "results/realdat_realmodel/"


dtst = int(sys.argv[1])# <--------------------------- ARG 1 -------------------------
idx_nl = int(sys.argv[2])# <--------------------------- ARG 2 -------------------------
gen_mt = int(sys.argv[3])#1# <--------------------------- ARG 3 -------------------------

type = "simple"; tp = "sg"
if gen_mt > 1:
    type = "complex"; tp = "cg"

dbfile = open('models_and_data/real_data_'+type+'.pkl', 'rb')
models = pickle.load(dbfile)
tranf_data = pickle.load(dbfile)
datasets = pickle.load(dbfile)
dbfile.close()

## RUN
filename = 'res_exp_realdat_real_model_gen_'+tp+'_d_'+str(dtst)+'_nl_'+str(nrls[idx_nl])+'.csv'

# We don't want to run it several times.
if os.path.isfile(path+filename):
    exit(0)


rseed = ((5 + dtst)*7 + idx_nl)*11
if type == "complex":
    rseed = rseed*17
np.random.seed(rseed)
random.seed(rseed)


real_model = models[dtst][idx_nl]
cardinalities = real_model.cardinalities
iClass = real_model.class_ind


gen_res_real_model = []

for d in np.arange(idx_nl*len(num_full_labeled_insts)*num_exp_reps_datasets,
                   (idx_nl+1)*len(num_full_labeled_insts)*num_exp_reps_datasets):
    print(dtst," ", nrls[idx_nl]," ", d)

    if len(datasets[dtst][d]) > 0:
        idxs_sample = datasets[dtst][d][0]
        dataset = tranf_data[dtst][idx_nl][idxs_sample,:]
        # first_l_cs = create_genModel_gost_candidate_sets(datasets[d][0], gen_model)
        res_real_model = only_test(real_model, cardinalities, iClass, dataset,
                                   num_cv_folds, num_cv_reps, showTrace)
        gen_res_real_model.append(res_real_model)

        for ind_wlp in np.arange(len(rel_prop_weak_labeled_insts)):
            print(ind_wlp)
            if len(datasets[dtst][d][ind_wlp+1]) > 0:
                idxs_sample = np.concatenate((idxs_sample,datasets[dtst][d][ind_wlp + 1]))
                idxs_sample.sort()
                dataset = tranf_data[dtst][idx_nl][idxs_sample, :]

                res_real_model = only_test(real_model, cardinalities, iClass, dataset,
                                           num_cv_folds, num_cv_reps, showTrace)
                gen_res_real_model.append(res_real_model)
            else:
                gen_res_real_model.append(-np.ones(5)*np.inf)
                print("no weak dataset")

    else:
        for i in np.arange(5):
            gen_res_real_model.append(-np.ones(5) * np.inf)
        print("no full dataset")


np.savetxt(path + filename,
           np.array(gen_res_real_model), delimiter=",", fmt='%.4e')
