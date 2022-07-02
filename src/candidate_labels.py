import copy

import numpy as np
from itertools import product

def create_random_candidate_sets(labels, class_cardinality=6, cand_set_size=None):
    if cand_set_size is None:
        cand_set_size = class_cardinality // 2

    candidate_sets = []
    for i in np.arange(len(labels)):
        cand_set = np.random.choice(class_cardinality, cand_set_size, replace=False)
        if labels[i] not in cand_set:
            cand_set[np.random.choice(cand_set_size,1)] = labels[i]
        candidate_sets.append(np.sort(cand_set).astype(int))

    return candidate_sets




def create_confMatrix_candidate_sets_by_prob_distr(labels, class_cardinality=6, confMatrix=None, prob_set_size=None):
    if prob_set_size is None:
        prob_set_size = np.array([1 / class_cardinality]*class_cardinality)

    if confMatrix is None:
        confMatrix = np.ones((class_cardinality,class_cardinality))
        confMatrix *= 1.0 / class_cardinality

    candidate_sets = []
    for i in np.arange(len(labels)):
        cand_set_size = np.random.choice(np.arange(class_cardinality)+1, 1, p=prob_set_size)[0]
        #cand_set = np.random.choice(class_cardinality, cand_set_size, replace=False, p=confMatrix[int(labels[i]),:])

        cand_set = np.zeros(cand_set_size)
        cand_set[0] = labels[i]
        if cand_set_size > 1:
            act_pd_labels = confMatrix[int(labels[i]),:].copy()
            act_pd_labels[int(labels[i])] = 0
            act_pd_labels /= np.sum(act_pd_labels)
            cand_set[1:] = np.random.choice(class_cardinality, cand_set_size-1, replace=False, p=act_pd_labels)

        #if labels[i] not in cand_set:
        #    cand_set[np.random.choice(cand_set_size,1)] = labels[i]

        candidate_sets.append(np.sort(cand_set).astype(int))

    return candidate_sets


def create_genModel_candidate_sets_by_prob_distr(instances, model, class_cardinality=6,
                                                 prob_set_size=None):
    if prob_set_size is None:
        prob_set_size = np.array([1 / class_cardinality]*class_cardinality)

    candidate_sets = []
    for i in np.arange(instances.shape[0]):
        cand_set_size = np.random.choice(np.arange(class_cardinality)+1, 1, p=prob_set_size)[0]
        #cand_set = np.random.choice(class_cardinality, cand_set_size, replace=False, p=confMatrix[int(labels[i]),:])

        cand_set = np.zeros(cand_set_size)
        cand_set[0] = instances[i, model.class_ind]
        if cand_set_size > 1:
            act_pd_labels = model.probability_distribution(instances[i])
            act_pd_labels[int(cand_set[0])] = 0
            act_pd_labels /= np.sum(act_pd_labels)
            cand_set[1:] = np.random.choice(class_cardinality, cand_set_size-1, replace=False, p=act_pd_labels)

        #if labels[i] not in cand_set:
        #    cand_set[np.random.choice(cand_set_size,1)] = labels[i]

        candidate_sets.append(np.sort(cand_set).astype(int))

    return candidate_sets


def create_genModel_complex_candidate_sets_by_prob_distr(instances, model, class_cardinality=6,
                                                         prob_set_size=None, complexity=0.5):
    if prob_set_size is None:
        prob_set_size = np.array([1 / class_cardinality]*class_cardinality)

    candidate_sets = []
    for i in np.arange(instances.shape[0]):
        cand_set_size = np.random.choice(np.arange(class_cardinality)+1, 1, p=prob_set_size)[0]

        cand_set = np.zeros(cand_set_size)
        cand_set[0] = instances[i, model.class_ind]

        if cand_set_size > 1:
            act_pd_labels = model.probability_distribution(instances[i])
            act_pd_labels /= act_pd_labels[int(cand_set[0])]
            act_pd_labels[int(cand_set[0])] = 0
            ordered_labels = np.arange(class_cardinality)[np.argsort(act_pd_labels)]
            ordered_labels = ordered_labels[1:]
            first_ind = int(np.round((class_cardinality-cand_set_size)*complexity))

            cand_set[1:] = ordered_labels[first_ind:(first_ind+cand_set_size-1)]

        candidate_sets.append(np.sort(np.unique(cand_set)).astype(int))

    return candidate_sets


def create_genModel_complex_candidate_sets_fixed_size(instances, model, class_cardinality=6,
                                                         cand_set_size=2, complexity=0.5):

    if cand_set_size < 1 or cand_set_size > class_cardinality:
        print("CS size incorrect")
        exit()

    candidate_sets = []
    for i in np.arange(instances.shape[0]):

        cand_set = np.zeros(cand_set_size)
        cand_set[0] = instances[i, model.class_ind]

        if cand_set_size > 1:
            act_pd_labels = model.probability_distribution(instances[i])
            act_pd_labels /= act_pd_labels[int(cand_set[0])]
            act_pd_labels[int(cand_set[0])] = 0
            ordered_labels = np.arange(class_cardinality)[np.argsort(act_pd_labels)]
            ordered_labels = ordered_labels[1:]
            first_ind = int(np.round((class_cardinality-cand_set_size)*complexity))

            cand_set[1:] = ordered_labels[first_ind:(first_ind+cand_set_size-1)]

        candidate_sets.append(np.sort(np.unique(cand_set)).astype(int))

    return candidate_sets

def confusionMatrixFirstAndSecond(model, class_cardinality=6):
    cmprobs = np.ones((class_cardinality,class_cardinality))

    vals = [np.arange(c) for c in model.cardinalities]
    vals[model.class_ind] = np.array([0])
    for instance in product(*vals):
        probs = model.probability_distribution(np.asarray(instance))
        ordered_labels = np.argsort(-probs)
        cmprobs[ ordered_labels[0], ordered_labels[1] ] += 1
    for c in np.arange(class_cardinality):
        cmprobs[c,c] = 0
    return cmprobs

def create_genModel_complex_consistent_candidate_sets_fixed_size(instances, model, cmprobs, class_cardinality=6,
                                                                 cand_set_size=2, complexity=0.5, consistency=1.0):

    if cand_set_size < 1 or cand_set_size > class_cardinality:
        print("CS size incorrect")
        exit()


    first_ind = int(np.round((class_cardinality - cand_set_size) * complexity))

    masks=[]
    for c in np.arange(class_cardinality):
        ordered_labels = np.arange(class_cardinality)[np.argsort(cmprobs[c,:])]
        ordered_labels = ordered_labels[1:]
        masks.append(ordered_labels[first_ind:(first_ind+cand_set_size-1)])

    candidate_sets = []
    for i in np.arange(instances.shape[0]):

        cand_set = np.zeros(cand_set_size)
        cand_set[0] = instances[i, model.class_ind]

        if cand_set_size > 1:
            act_mask = masks[int(cand_set[0])]
            act_pd_labels = model.probability_distribution(instances[i])
            act_pd_labels[act_mask] = 1
            act_pd_labels[int(cand_set[0])] = 0
            act_pd_labels = np.exp(consistency*act_pd_labels)/np.sum(np.exp(consistency*act_pd_labels))
            act_pd_labels[int(cand_set[0])] = 0
            act_pd_labels /= np.sum(act_pd_labels)
            cand_set[1:] = np.random.choice(class_cardinality, size=cand_set_size-1, replace=False, p=act_pd_labels)

        candidate_sets.append(np.sort(np.unique(cand_set)).astype(int))

    return candidate_sets

# def create_genModel_complex_candidate_sets_by_prob_distr(instances, model, class_cardinality=6,
#                                                          proportion_of_labeled=0.5,
#                                                          complexity=1.0):
#     candidate_sets = []
#     for i in np.arange(instances.shape[0]):
#         val = np.random.choice(2,1,p=np.array([proportion_of_labeled,1-proportion_of_labeled]))
#
#         real_label = instances[i, model.class_ind]
#
#         if val < .5:
#             cand_set = np.array([real_label])
#         else:
#             act_pd_labels = model.probability_distribution(instances[i])
#             act_pd_labels[int(real_label)] = 0
#             act_pd_labels /= np.sum(act_pd_labels)
#             ord = np.argsort(act_pd_labels)
#             ordered_labels = np.flip(np.arange(class_cardinality)[ord])
#             cum_pd_labels = np.cumsum(np.flip(act_pd_labels[ord]))
#             ind = np.where(cum_pd_labels >= complexity)[0][0]
#             cand_set = ordered_labels[:(ind+1)]
#             np.insert(cand_set,0,real_label)
#
#         candidate_sets.append(np.sort(np.unique(cand_set)).astype(int))
#
#     return candidate_sets



def create_genModel_gost_candidate_sets(instances, model):

    candidate_sets = []
    for i in np.arange(instances.shape[0]):
        cand_set = np.zeros(1)
        cand_set[0] = instances[i, model.class_ind]
        candidate_sets.append(cand_set.astype(int))

    return candidate_sets

# scenario = 0 (honest); 1 (misleading)
def new_create_genModel_complex_candidate_sets_fixed_size(instances, model, class_cardinality=6,
                                                         cand_set_size=2, scenario=0, sampling=1,
                                                          cmprobs=None, num_consistent_labels=0, strength=None
                                                          ):

    if cand_set_size < 2 or cand_set_size > class_cardinality:
        print("CS size incorrect")
        exit()
    masks = []

    if 0 < num_consistent_labels:
        if num_consistent_labels != 1:
            print("only 1 consistent label implemented-allowed")
            exit()

        # cmprobs = (cmprobs + np.transpose(cmprobs) )/2 # we make it symmetric CAL?¿??
        for c in np.arange(class_cardinality):
            ordered_labels = np.arange(class_cardinality)[np.argsort(cmprobs[c, :])]
            masks.append(ordered_labels[1:])

    candidate_sets = []
    for i in np.arange(instances.shape[0]):
        cand_set = np.zeros(cand_set_size)
        cand_set[0] = instances[i, model.class_ind]
        n_already_fix_labs = 1

        act_pd_labels = model.probability_distribution(instances[i])
        act_pd_labels[int(cand_set[0])] = 0
        act_pd_labels /= np.sum(act_pd_labels)
        act2_pd_labels = copy.deepcopy(act_pd_labels)

        if scenario > 0:
            ## OPTION 1: swap order of labels, keep same probabilities
            #min_to_max = np.argsort(act_pd_labels)[1:]
            #act2_pd_labels[min_to_max[::-1]] = act_pd_labels[min_to_max]
            ## OPTION 2: inversely proportional probability
            act2_pd_labels = (1 - act2_pd_labels)
            act2_pd_labels[int(cand_set[0])] = 0
            act2_pd_labels /= np.sum(act2_pd_labels)

        if num_consistent_labels > 0:
            act_mask = masks[int(cand_set[0])]
            if np.random.random_sample() < strength:
                cand_set[1] = act_mask[0]
                n_already_fix_labs += 1
                act2_pd_labels[int(cand_set[1])] = 0
                act2_pd_labels /= np.sum(act2_pd_labels)

        if sampling == 1:
            cand_set[n_already_fix_labs:] = np.random.choice(class_cardinality, cand_set_size-n_already_fix_labs,
                                                                 p=act2_pd_labels, replace=False)
        else:
            ordered_labels = np.arange(class_cardinality)[np.argsort(act2_pd_labels)[::-1]]
            cand_set[n_already_fix_labs:] = ordered_labels[:(cand_set_size-n_already_fix_labs)]

        candidate_sets.append(np.sort(np.unique(cand_set)).astype(int))

    return candidate_sets
