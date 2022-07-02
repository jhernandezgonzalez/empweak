from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score,accuracy_score,log_loss
from EM import *
from bnc import *
from auxiliary import *


def experiment_Weak_and_Full(model_ini, cardinalities, iClass, dataset, cand_sets, num_folds, num_reps, showTrace):

    learning = EM(model_ini, iClass, cardinalities, verbose=showTrace)

    results_weak = []
    results_full = []

    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_reps)
    for i, (train_index, test_index) in enumerate(rkf.split(dataset)):
        if showTrace:
            print("Iteration CV",i)
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        cand_sets_train = [cand_sets[i] for i in train_index]

        # LEARNING
        # a) weakly supervised classifier
        model, weights = learning.run(dataset_train, cand_sets_train)

        # b) fully supervised classifier
        full_model = copy.deepcopy(model_ini)
        full_model.clear()
        full_model.learn( filter_weak_out(dataset_train, cand_sets_train) )

        # VALIDATION
        # a) weakly supervised classifier
        pred_labels = model.classify_set(dataset_test)
        pred_probs = model.probability_distribution_set(dataset_test)
        pred_probs = np.array(pred_probs)

        val_acc = accuracy_score(dataset_test[:,iClass], pred_labels)
        val_macrof1 = f1_score(dataset_test[:,iClass], pred_labels, average='macro')
        val_microf1 = f1_score(dataset_test[:,iClass], pred_labels, average='micro')
        val_log_loss = log_loss(dataset_test[:,iClass], pred_probs, labels=np.arange(cardinalities[iClass]))
        val_brier_sc = brier_score(dataset_test[:,iClass], pred_probs)

        results_weak.append([val_acc, val_macrof1, val_microf1, val_log_loss, val_brier_sc])

        # b) fully supervised classifier
        pred_labels = full_model.classify_set(dataset_test)
        pred_probs = full_model.probability_distribution_set(dataset_test)
        pred_probs = np.array(pred_probs)

        val_acc = accuracy_score(dataset_test[:,iClass], pred_labels)
        val_macrof1 = f1_score(dataset_test[:,iClass], pred_labels, average='macro')
        val_microf1 = f1_score(dataset_test[:,iClass], pred_labels, average='micro')
        val_log_loss = log_loss(dataset_test[:,iClass], pred_probs, labels=np.arange(cardinalities[iClass]))
        val_brier_sc = brier_score(dataset_test[:,iClass], pred_probs)
        results_full.append([val_acc, val_macrof1, val_microf1, val_log_loss, val_brier_sc])

    return np.mean(results_weak, 0), np.mean(results_full, 0)


def experiment_only_Full(model_ini, cardinalities, iClass, dataset, num_folds, num_reps, showTrace):

    results_full = []

    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_reps)
    for i, (train_index, test_index) in enumerate(rkf.split(dataset)):
        if showTrace:
            print("Iteration CV",i)
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]

        # LEARNING
        # b) fully supervised classifier
        full_model = copy.deepcopy(model_ini)
        full_model.clear()
        full_model.learn( dataset_train )

        # VALIDATION

        # b) fully supervised classifier
        pred_labels = full_model.classify_set(dataset_test)
        pred_probs = full_model.probability_distribution_set(dataset_test)
        pred_probs = np.array(pred_probs)

        val_acc = accuracy_score(dataset_test[:,iClass], pred_labels)
        val_macrof1 = f1_score(dataset_test[:,iClass], pred_labels, average='macro')
        val_microf1 = f1_score(dataset_test[:,iClass], pred_labels, average='micro')
        val_log_loss = log_loss(dataset_test[:,iClass], pred_probs, labels=np.arange(cardinalities[iClass]))
        val_brier_sc = brier_score(dataset_test[:,iClass], pred_probs)
        results_full.append([val_acc, val_macrof1, val_microf1, val_log_loss, val_brier_sc])

    return np.mean(results_full, 0)


def only_test(model, cardinalities, iClass, dataset, num_folds, num_reps, showTrace):

    results_full = []

    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_reps)
    for i, (train_index, test_index) in enumerate(rkf.split(dataset)):
        if showTrace:
            print("Iteration CV",i)
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]

        # VALIDATION

        # b) fully supervised classifier
        pred_labels = model.classify_set(dataset_test)
        pred_probs = model.probability_distribution_set(dataset_test)
        pred_probs = np.array(pred_probs)

        val_acc = accuracy_score(dataset_test[:,iClass], pred_labels)
        val_macrof1 = f1_score(dataset_test[:,iClass], pred_labels, average='macro')
        val_microf1 = f1_score(dataset_test[:,iClass], pred_labels, average='micro')
        val_log_loss = log_loss(dataset_test[:,iClass], pred_probs, labels=np.arange(cardinalities[iClass]))
        val_brier_sc = brier_score(dataset_test[:,iClass], pred_probs)
        results_full.append([val_acc, val_macrof1, val_microf1, val_log_loss, val_brier_sc])

    return np.mean(results_full, 0)


def experiment_only_Weak(model_ini, cardinalities, iClass, dataset, cand_sets, num_folds, num_reps, showTrace):

    learning = EM(model_ini, iClass, cardinalities, verbose=showTrace)

    results_weak = []

    rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_reps)
    for i, (train_index, test_index) in enumerate(rkf.split(dataset)):
        if showTrace:
            print("Iteration CV",i)
        dataset_train, dataset_test = dataset[train_index], dataset[test_index]
        cand_sets_train = [cand_sets[i] for i in train_index]

        # LEARNING
        # a) weakly supervised classifier
        model, weights = learning.run(dataset_train, cand_sets_train)

        # VALIDATION
        # a) weakly supervised classifier
        pred_labels = model.classify_set(dataset_test)
        pred_probs = model.probability_distribution_set(dataset_test)
        pred_probs = np.array(pred_probs)

        val_acc = accuracy_score(dataset_test[:,iClass], pred_labels)
        val_macrof1 = f1_score(dataset_test[:,iClass], pred_labels, average='macro')
        val_microf1 = f1_score(dataset_test[:,iClass], pred_labels, average='micro')
        val_log_loss = log_loss(dataset_test[:,iClass], pred_probs, labels=np.arange(cardinalities[iClass]))
        val_brier_sc = brier_score(dataset_test[:,iClass], pred_probs)

        results_weak.append([val_acc, val_macrof1, val_microf1, val_log_loss, val_brier_sc])

    return np.mean(results_weak,0)


def filter_weak_out(dataset, cand_sets):
    full_labeled = [len(cs) == 1 for cs in cand_sets]
    return dataset[full_labeled,:]


def real_error(gen_model, learn_model):
    vals = [np.arange(c) for c in gen_model.cardinalities]
    vals[gen_model.class_ind] = np.array([0])
    dataset = []
    weights = []
    for instance in product(*vals):
        dataset.append(np.asarray(instance))
        weights.append(gen_model.probability_distribution(np.asarray(instance)))

    dataset = np.array(dataset)
    weights = np.array(weights)

    learn_model.learn(dataset, weights, smoothing=0)

    err = 0.0
    for inst_ind in np.arange(len(dataset)):
        probs = learn_model.probability_distribution(dataset[inst_ind, :])
        err += 1.0 - probs[ np.argmax(weights[inst_ind, :]) ]

    return err / len(dataset)
