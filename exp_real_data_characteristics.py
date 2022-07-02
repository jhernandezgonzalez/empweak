import numpy as np
import sys
import pickle
import warnings

from src.bnc import KDB

warnings.filterwarnings('ignore')

## SETTINGS

showTrace = False

kDB_gen = int(sys.argv[1])# <--------------------------- ARG 1 -------------------------

complexity = "simple" # if kDB_gen = 1
if kDB_gen == 4:
    complexity = "complex"


num_full_labeled_insts = np.round((1 + np.arange(10)) * (100.0 / 3)).astype(int)
rel_prop_weak_labeled_insts = np.array([0.5, 1, 2, 5])
num_exp_reps_datasets = 30
nrls = np.arange(4,9)


def measure_meaningfulness(cand_set, real_lab, distr):
    n_labels = len(distr)
    cs_size = len(cand_set) - 1

    distr[real_lab] = -np.inf
    ord_labs = np.argsort(distr)

    act_m = np.mean([np.where(ord_labs == cl)[0][0] for cl in cand_set if cl != real_lab])
    min_m = np.mean(np.arange(cs_size) + 1)
    max_m = np.mean(np.arange(n_labels - cs_size, n_labels))
    if max_m - min_m == 0: return 0
    return (act_m - min_m) / (max_m - min_m)


def measure_ambiguity(cand_sets, y, nlabs):
    amb = []
    for c in np.arange(nlabs):
        cand_labs = np.zeros(nlabs)
        for act_y, cs in zip(y,cand_sets):
            if act_y == c:
                cand_labs[cs] += 1

        cand_labs /= np.sum(y == c)
        cand_labs[c] = 0
        amb.append(np.max(cand_labs))
    amb = np.array(amb)
    return amb.mean()



# read data
dbfile = open('models_and_data/real_data_' + complexity + '.pkl', 'rb')
orig_models = pickle.load(dbfile)
orig_transf_data = pickle.load(dbfile)
orig_datasets = pickle.load(dbfile)
orig_transf_csets = pickle.load(dbfile)
dbfile.close()



# output pickle
dbfile = open('models_and_data/real_data_characteristics_' + complexity + '.pkl', 'wb')

transdtst_meaningfullness = np.zeros((3, 5))
transdtst_ambiguity = np.zeros((3, 5))
transdtst_real_cs_mean_size = np.zeros((3, 5))
transdtst_csld_cs_mean_size = np.zeros((3, 5))

for dtst in np.arange(3):
    print("================================================================")

    for i_nrl, nrl in enumerate(np.arange(4, 9)):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(dtst, nrl)
        model = orig_models[dtst][i_nrl]
        iclass = model.class_ind

        dataset = orig_transf_data[dtst][i_nrl]
        candsets = orig_transf_csets[dtst][i_nrl]

        cs_sizes = np.array([len(cs) for cs in candsets])
        csld = np.where(cs_sizes > 1)[0]
        really_candsets = [candsets[i] for i in csld]
        #print(len(cs_sizes), len(csld), len(really_candsets))

        y_csld = dataset[csld, iclass]

        transdtst_real_cs_mean_size[dtst, i_nrl] = cs_sizes.mean()
        transdtst_csld_cs_mean_size[dtst, i_nrl] = np.mean(cs_sizes[cs_sizes > 1])

        transdtst_ambiguity[dtst, i_nrl] = measure_ambiguity(really_candsets, y_csld, nrl)

        transdtst_meaningfullness[dtst, i_nrl] = np.mean([measure_meaningfulness(candsets[i], dataset[i, iclass],
                                                                                 model.probability_distribution(
                                                                                     dataset[i, :]))
                                                          for i in csld])
        # print(dataset.shape[0], len(csld), transdtst_meaningfullness[dtst,i_nrl])

print(transdtst_real_cs_mean_size)
print(transdtst_csld_cs_mean_size)
print(transdtst_meaningfullness)
print(transdtst_ambiguity)

pickle.dump(transdtst_meaningfullness, dbfile)
pickle.dump(transdtst_ambiguity, dbfile)
pickle.dump(transdtst_real_cs_mean_size, dbfile)
pickle.dump(transdtst_csld_cs_mean_size, dbfile)






useddtst_meaningfullness = []
useddtst_ambiguity = []
useddtst_real_cs_mean_size = []
useddtst_csld_cs_mean_size = []

for dtst in np.arange(3):
    print("================================================================")

    idx_act_dst = -1
    for i_nrl, nrl in enumerate(np.arange(4, 9)):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(dtst, nrl)
        model = orig_models[dtst][i_nrl]
        iclass = model.class_ind

        submat_real_cs = -1 * np.ones(
            (num_exp_reps_datasets, len(num_full_labeled_insts), len(rel_prop_weak_labeled_insts)))
        submat_csld_cs = -1 * np.ones(
            (num_exp_reps_datasets, len(num_full_labeled_insts), len(rel_prop_weak_labeled_insts)))
        submat_meaning = -1 * np.ones(
            (num_exp_reps_datasets, len(num_full_labeled_insts), len(rel_prop_weak_labeled_insts)))
        submat_ambiguity = -1 * np.ones(
            (num_exp_reps_datasets, len(num_full_labeled_insts), len(rel_prop_weak_labeled_insts)))

        for d in np.arange(num_exp_reps_datasets):
            for ind_nfi in np.arange(len(num_full_labeled_insts)):
                idx_act_dst += 1
                act_datasets = orig_datasets[dtst][idx_act_dst]
                if len(act_datasets) == 0:
                    continue;

                idxs_sample = act_datasets[0]
                for ind_wlp in np.arange(len(rel_prop_weak_labeled_insts)):
                    if len(act_datasets[ind_wlp + 1]) == 0:
                        continue;
                    idxs_sample = np.concatenate((idxs_sample, act_datasets[ind_wlp + 1]))
                    dataset = orig_transf_data[dtst][i_nrl][idxs_sample, :]
                    candsets = [orig_transf_csets[dtst][i_nrl][i] for i in idxs_sample]

                    cs_sizes = np.array([len(cs) for cs in candsets])
                    csld = np.where(cs_sizes > 1)[0]
                    really_candsets = [candsets[i] for i in csld]
                    y_csld = dataset[csld, iclass]

                    submat_real_cs[d, ind_nfi, ind_wlp] = cs_sizes.mean()
                    submat_csld_cs[d, ind_nfi, ind_wlp] = np.mean(cs_sizes[cs_sizes > 1])

                    submat_ambiguity[d, ind_nfi, ind_wlp] = measure_ambiguity(really_candsets, y_csld, nrl)

                    submat_meaning[d, ind_nfi, ind_wlp] = np.mean(
                        [measure_meaningfulness(candsets[i], dataset[i, iclass],
                                                model.probability_distribution(dataset[i, :]))
                         for i in csld])
        aux = np.mean(submat_real_cs, axis=0)
        print(aux)
        useddtst_real_cs_mean_size.append(aux)
        aux = np.mean(submat_csld_cs, axis=0)
        print(aux)
        useddtst_csld_cs_mean_size.append(aux)
        aux = np.mean(submat_meaning, axis=0)
        print(aux)
        useddtst_meaningfullness.append(aux)
        aux = np.mean(submat_ambiguity, axis=0)
        print(aux)
        useddtst_ambiguity.append(aux)

pickle.dump(useddtst_meaningfullness, dbfile)
pickle.dump(useddtst_ambiguity, dbfile)
pickle.dump(useddtst_real_cs_mean_size, dbfile)
pickle.dump(useddtst_csld_cs_mean_size, dbfile)

dbfile.close()
