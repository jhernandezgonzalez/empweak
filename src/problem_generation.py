import numpy as np


def create_problem(n_vars=10, cardinalities=2, class_cardinality=6, n_parents=2, alpha_0=1):
    if type(cardinalities) == int:
        cardinalities = [cardinalities] * n_vars
    elif len(cardinalities) != n_vars:
        print("The vector of cardinalities do not match the number of predictive variables")
        exit()
    cardinalities.insert(0,class_cardinality)# = [class_cardinality, cardinalities].flatten
    cardinalities = np.array(cardinalities)
    parents = [None] * (n_vars+1)
    prob_distr = [None] * (n_vars+1)
    # the class variable has no parents
    # parents[0] = []
    prob_distr[0] = np.random.dirichlet([alpha_0]*class_cardinality, 1)

    ordering = np.random.choice(np.arange(n_vars)+1,n_vars,False)
    for i in np.arange(n_vars):
        cand_parents = ordering[:i]
        if len(cand_parents) > n_parents:
            cand_parents = np.random.choice(cand_parents, n_parents, False)
        cand_parents = np.append(cand_parents,0) # the class var is always a parent
        cand_parents = np.sort(cand_parents)
        par_joint_card = np.prod(cardinalities[cand_parents])

        #cond_prob_distrs = np.zeros((cardinalities[ordering[i]], par_joint_card))
        cond_prob_distrs = np.random.dirichlet([alpha_0] * cardinalities[ordering[i]],
                                               par_joint_card)

        parents[ordering[i]] = cand_parents
        prob_distr[ordering[i]] = cond_prob_distrs

    return cardinalities, ordering, parents, prob_distr


def create_dataset_from_problem(problem, n_samples=100):
    return create_dataset(problem[0], problem[1], problem[2], problem[3], n_samples)


def create_dataset(cardinalities, ordering, parents, prob_distr, n_samples=100):
    dataset = np.zeros((n_samples,len(ordering)+1))
    for n in np.arange(n_samples):
        instance = np.zeros(len(ordering)+1)

        instance[0] = np.where(np.random.multinomial(1, prob_distr[0][0,:], 1) == 1)[1]

        for v in ordering:
            prev_card = 1
            par_index = 0
            for p in parents[v]:
                par_index += instance[p] * prev_card
                prev_card *= cardinalities[p]
            instance[v] = np.where(np.random.multinomial(1,prob_distr[v][int(par_index), :],1) == 1)[1]

        dataset[n,:] = instance

    return dataset.astype(int)