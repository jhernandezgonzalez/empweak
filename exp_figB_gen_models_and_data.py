import pickle
import random
import numpy as np

from src.bnc import KDB

import warnings
warnings.filterwarnings('ignore')

## SETTINGS

path = "models_and_data/"
num_vars = 16
cards = 2
card_class = 6
alpha_0 = 1
kDB_gen = 1

num_full_labeled_insts = np.round((1+np.arange(10))*(100.0/3)).astype(int)
rel_prop_weak_labeled_insts = np.array([0.5, 1, 2, 5])

num_exp_reps_datasets = 30
num_exp_reps_models = 30

# create data cardinalities
cardinalities = [cards] * num_vars
cardinalities.append(card_class)
cardinalities = np.array(cardinalities)
iClass = len(cardinalities) - 1


mt = "simplegen"
#mt = "complexgen"

dbfile = open(path+'figA_'+mt+'.pkl', 'rb')

models = pickle.load(dbfile)
conf_mtrcs = pickle.load(dbfile)

dbfile.close()


rseed = 31
np.random.seed(rseed)
random.seed(rseed)

dbfile = open(path+'figB_'+mt+'.pkl', 'wb')
pickle.dump(models, dbfile)
pickle.dump(conf_mtrcs, dbfile)

datasets = []
for m in np.arange(num_exp_reps_models):
    for d in np.arange(num_exp_reps_datasets):
        for num_full_insts in num_full_labeled_insts:
            # create sizes of the subsets of data
            l_num_weak_insts = np.round(num_full_insts * rel_prop_weak_labeled_insts).astype(int)

            l_num_insts_per_dataset = np.delete(l_num_weak_insts, len(l_num_weak_insts) - 1)
            l_num_insts_per_dataset = np.insert(l_num_insts_per_dataset, 0, 0)
            l_num_insts_per_dataset = l_num_weak_insts - l_num_insts_per_dataset
            l_num_insts_per_dataset = np.insert(l_num_insts_per_dataset, 0, num_full_insts)

            print(m, " ", d, " ", num_full_insts)
            data = models[m].create_datasets(l_num_insts_per_dataset)
            datasets.append(data)

pickle.dump(datasets, dbfile)

dbfile.close()
