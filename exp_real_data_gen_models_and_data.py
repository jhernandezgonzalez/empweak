import pickle
import random
import warnings
import numpy as np

from src.auxiliary import indx_sample_stratified
from src.bnc import KDB

warnings.filterwarnings('ignore')

## SETTINGS

path = "models_and_data/"
avoid_first = [0,1,0]

dbfile = open(path + 'real_data.pkl', 'rb')
data = pickle.load(dbfile)
dbfile.close()

kDB_gen = 1; dbfile = open(path+'real_data_simple.pkl', 'wb')
#kDB_gen = 4; dbfile = open(path+'real_data_complex.pkl', 'wb')


num_full_labeled_insts = np.round((1+np.arange(10))*(100.0/3)).astype(int)
rel_prop_weak_labeled_insts = np.array([0.5, 1, 2, 5])
num_exp_reps_datasets = 30


rseed = 17
np.random.seed(rseed)
random.seed(rseed)

models = []
transf_data = []
transf_cand_sets = []
#conf_mtrcs = []
datasets = []

for dtst in np.arange(3):
    print("================================================================")
    models_per_realdat=[]
    tdata_per_realdat=[]
    tcsets_per_realdat=[]
    #cm_per_realdat=[]
    datasets_per_realdat=[]
    for nrl in np.arange(4,9):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        data_new = data[dtst].subset(nrl,avoid_first[dtst])
        data_new.binarize()
        data_new.remove_no_info()
        data_new.remove_redundant_corr()

        cur_dataset, cur_candsets, iClass, cardinalities = data_new.return_dataset()
        tdata_per_realdat.append(cur_dataset)
        tcsets_per_realdat.append(cur_candsets)

        print("Learn")
        gen_model = KDB(iClass, cardinalities, K=kDB_gen)
        gen_model.structural_learning(cur_dataset)
        gen_model.learn(cur_dataset)

        #print("CM")
        #cmprobs = confusionMatrixFirstAndSecond(gen_model)

        models_per_realdat.append(gen_model)
        #cm_per_realdat.append(cmprobs)

        cs_sizes = np.sum(data_new.cand_sets, axis=1)
        full_lab = np.arange(len(cs_sizes))[cs_sizes == 1]
        full_real_lab = np.where(data_new.y[full_lab, :] == 1)[1]
        cs_lab = np.arange(len(cs_sizes))[cs_sizes > 1]
        cs_real_lab = np.where(data_new.y[cs_lab, :] == 1)[1]

        print(len(full_lab), len(cs_lab))

        # create sizes of the subsets of data
        for d in np.arange(num_exp_reps_datasets):
            for num_full_insts in num_full_labeled_insts:
                print(dtst, nrl, d, num_full_insts)

                l_num_weak_insts = np.round(num_full_insts * rel_prop_weak_labeled_insts).astype(int)

                l_num_insts_per_dataset = np.delete(l_num_weak_insts, len(l_num_weak_insts) - 1)
                l_num_insts_per_dataset = np.insert(l_num_insts_per_dataset, 0, 0)
                l_num_insts_per_dataset = l_num_weak_insts - l_num_insts_per_dataset

                #print(l_num_insts_per_dataset)
                # CHECK, hay suficientes en el dataset original??
                act_dataset = []
                if num_full_insts < len(full_lab):
                    # mesurar meaningfulness
                    full_subset = indx_sample_stratified(full_lab, full_real_lab, num_full_insts)
                    act_dataset.append(full_subset)

                    run_cs_lab = cs_lab.copy()
                    run_cs_real_lab = cs_real_lab.copy()

                    for num_weak_insts in l_num_insts_per_dataset:
                        #print("   ",num_weak_insts, len(run_cs_lab), len(run_cs_real_lab))
                        if num_weak_insts > len(run_cs_lab):
                            act_dataset.append([])
                        else:
                            #print(num_weak_insts, len(run_cs_lab), len(run_cs_real_lab))
                            weak_subset = indx_sample_stratified(run_cs_lab, run_cs_real_lab, num_weak_insts)
                            act_dataset.append(weak_subset)
                            unselected = [i for i in np.arange(len(run_cs_lab)) if run_cs_lab[i] not in weak_subset]
                            run_cs_lab = run_cs_lab[unselected]
                            run_cs_real_lab = run_cs_real_lab[unselected]

                datasets_per_realdat.append(act_dataset)

    models.append(models_per_realdat)
    transf_data.append(tdata_per_realdat)
    transf_cand_sets.append(tcsets_per_realdat)
    #conf_mtrcs.append(cm_per_realdat)
    datasets.append(datasets_per_realdat)

pickle.dump(models, dbfile)
pickle.dump(transf_data, dbfile)
#pickle.dump(conf_mtrcs, dbfile)
pickle.dump(datasets, dbfile)
pickle.dump(transf_cand_sets, dbfile)

dbfile.close()
