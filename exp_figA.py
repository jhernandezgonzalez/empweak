import os
import pickle
import random
import numpy as np
import copy
import sys
import warnings

from src.bnc import KDB
from src.candidate_labels import create_genModel_gost_candidate_sets, \
    new_create_genModel_complex_candidate_sets_fixed_size
from src.experiments import experiment_only_Weak

warnings.filterwarnings('ignore')

## SETTINGS

num_cv_folds = 3
num_cv_reps = 5

showTrace = False


num_model = int(sys.argv[1])# <--------------------------- ARG 1 -------------------------
num_dataset = int(sys.argv[2])# <--------------------------- ARG 2 -------------------------

kDB_learn = 1
gen_mt = int(sys.argv[3])#1# <--------------------------- ARG 3 -------------------------

cs_size = int(sys.argv[4]) # <--------------------------- ARG 4 -------------------------
scene = int(sys.argv[5]) # <--------------------------- ARG 5 -------------------------

sampling = 1 #1: sampling; 0:deterministic
path = "results/figA_sampling/"
num_repetitions = 5

if sampling != 1:
    path = "results/figA_determ/"
    num_repetitions = 1


filename = 'res_exp_figA_m_' + str(num_model) + '_d_' + str(num_dataset) + '_css_' + str(cs_size) + \
           '_k_' + str(gen_mt) + '_s_' + str(scene) + '.csv'

# We don't want to run it several times.
if os.path.isfile(path+filename):
    sys.exit(0)

gen_mt_name = "figA_simplegen.pkl"
if gen_mt > 1:
    gen_mt_name = "figA_complexgen.pkl"
dbfile = open('models_and_data/'+gen_mt_name, 'rb')

orig_models = pickle.load(dbfile)
orig_conf_mtrcs = pickle.load(dbfile)
orig_datasets = pickle.load(dbfile)

dbfile.close()

num_orig_models = len(orig_models)
num_orig_datasets = int(len(orig_datasets) / num_orig_models)

## RUN
rseed = ((((5 + num_model)*7 + num_dataset)*11 + cs_size)*13 + kDB_learn)*17+scene
np.random.seed(rseed)
random.seed(rseed)

gen_model = orig_models[num_model]

cardinalities = gen_model.cardinalities
iClass = gen_model.class_ind

learn_model = KDB(iClass, cardinalities, K=kDB_learn)
learn_model.random_generation() # just to obtain the default kDB structure
learn_model.clear()

gen_res_weak = []

datasets = orig_datasets[num_model*num_orig_datasets+num_dataset]

for r in np.arange(num_repetitions):
    l_cand_sets = [create_genModel_gost_candidate_sets(datasets[0], gen_model)]
    for d in np.arange(1,len(datasets)):
        l_cand_sets.append(new_create_genModel_complex_candidate_sets_fixed_size(datasets[d],
                                                                                 gen_model,
                                                                                 cand_set_size=cs_size,
                                                                                 sampling=sampling,
                                                                                 scenario=scene))

    dataset = copy.deepcopy(datasets[0])
    cand_sets = copy.deepcopy(l_cand_sets[0])
    for ind_wlp in np.arange(len(datasets)-1):
        print(r, ind_wlp)
        dataset = np.vstack((dataset, copy.deepcopy(datasets[ind_wlp+1])))
        cand_sets.extend( copy.deepcopy(l_cand_sets[ind_wlp+1]) )

        res_weak = experiment_only_Weak(learn_model, cardinalities, iClass, dataset, cand_sets,
                                        num_cv_folds, num_cv_reps, showTrace)

        gen_res_weak.append( res_weak )

np.savetxt(path+filename,
           np.array(gen_res_weak), delimiter=",", fmt='%.4e')
