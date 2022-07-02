import pickle
import random
import numpy as np
import copy
import warnings

from src.experiments import only_test
from src.bnc import KDB

warnings.filterwarnings('ignore')

## SETTINGS

path = "results/"

gen_model=1

filename_genmodels = 'models_and_data/figB_simplegen.pkl'
if gen_model == 1:
    filename_genmodels = 'models_and_data/figB_complexgen.pkl'

dbfile = open(filename_genmodels, 'rb')

models = pickle.load(dbfile)
conf_mtrcs = pickle.load(dbfile)
datasets = pickle.load(dbfile)

dbfile.close()

num_cv_folds = 3
num_cv_reps = 5

showTrace = False


## RUN

filename = 'res_exp_figB_sg_real_model.resout'
if gen_model == 1:
    filename = 'res_exp_figB_cg_real_model.resout'

num_full_ds = 10
num_models = len(models)
num_datasets = int(len(datasets) / num_models / num_full_ds)

rseed = ((5 + num_models)*7 + num_datasets)*11 + gen_model
np.random.seed(rseed)
random.seed(rseed)






gen_res_real_model = []
for m in np.arange(num_models):
    real_model = models[m]
    cardinalities = real_model.cardinalities
    iClass = real_model.class_ind

    for d in np.arange(m*num_datasets*num_full_ds,(m+1)*num_datasets*num_full_ds):
        print(m," ", d)
        dataset = copy.deepcopy(datasets[d][0])

        res_real_model = only_test(real_model, cardinalities, iClass, dataset,
                                            num_cv_folds, num_cv_reps, showTrace)
        gen_res_real_model.append(res_real_model)

        for ind_wlp in np.arange(len(datasets[d]) - 1):
            print(ind_wlp)
            dataset = np.vstack((dataset, copy.deepcopy(datasets[d][ind_wlp + 1])))

            res_real_model = only_test(real_model, cardinalities, iClass, dataset,
                                                num_cv_folds, num_cv_reps, showTrace)
            gen_res_real_model.append(res_real_model)

np.savetxt(path + filename,
           np.array(gen_res_real_model), delimiter=",", fmt='%.4e')
