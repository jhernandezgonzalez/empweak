import numpy as np
from abc import ABCMeta, abstractmethod
from itertools import product

from src.auxiliary import obtain_full_weights, mutual_informations


class BNC(metaclass=ABCMeta):

    def __init__(self, class_ind, cardinalities, ordering=None):
        self.class_ind = class_ind
        self.cardinalities = cardinalities.copy()
        self.ordering = ordering
        if ordering is None:
            self.ordering = np.random.choice(len(self.cardinalities), len(self.cardinalities), False)
            ind = np.where(self.ordering == self.class_ind)[0]
            self.ordering[1:] = np.delete(self.ordering, ind)
            self.ordering[0] = self.class_ind
            self.ordering = self.ordering.astype(int)
        #print(self.ordering)
        self.parents = [None] * len(self.cardinalities)
        self.prob_distr = [None] * len(self.cardinalities)
        #self.clear()

    def clear(self):
        for prdistr in self.prob_distr:
            prdistr[:,:] = 0

    @abstractmethod
    def learn(self, dataset, weights=None, smoothing=1):
        pass

    def probability_distribution(self, instance):
        probs = self.prob_distr[self.class_ind][:,0].copy()
        for class_value in np.arange(self.cardinalities[self.class_ind]):
            aux_instance = instance.copy()
            aux_instance[self.class_ind] = class_value
            for v in self.ordering:
                if v != self.class_ind:
                    probs[class_value] *= self.prob_distr[v][aux_instance[v],
                                                             self.parent_joint_index(aux_instance,v)]

        return probs/np.sum(probs)

    def classify(self, instance):
        probs = self.probability_distribution(instance)
        return np.argmax(probs)

    def probability_distribution_set(self, dataset):
        probs = [self.probability_distribution(dataset[n, :])
                  for n in np.arange(dataset.shape[0])]
        return probs

    def classify_set(self, dataset):
        labels = [np.argmax(self.probability_distribution(dataset[n, :]))
                  for n in np.arange(dataset.shape[0])]
        return labels

    @abstractmethod
    def random_generation(self, alpha_0=1):
        pass

    def compare_to(self, model_b):
        result = 0 # np.sum((self.Pc - modelB.Pc) ** 2)

        for v in self.ordering:
            result += np.sum((self.prob_distr[v] - model_b.prob_distr[v]) ** 2)

        return np.sqrt(result)

    def create_dataset(self, n_samples=100):
        dataset = np.zeros((n_samples, len(self.ordering)))
        for n in np.arange(n_samples):
            instance = np.zeros(len(self.ordering))

            #instance[self.class_ind] = np.random.choice(self.cardinalities[self.class_ind], 1,
            #                                         p=self.prob_distr[self.class_ind][0, :])

            for v in self.ordering:
                par_joint_ind = self.parent_joint_index(instance, v)
                instance[v] = np.random.choice(self.cardinalities[v], 1,
                                               p=self.prob_distr[v][:, par_joint_ind])

            dataset[n, :] = instance

        return dataset.astype(int)

    def create_datasets(self, l_n_samples):
        datasets = []
        for s in l_n_samples:
            datasets.append(self.create_dataset(s))
        return datasets

    def parent_joint_index(self, instance, v):
        prev_card = 1
        par_index = 0
        if self.parents[v] is not None:
            for p in self.parents[v]:
                par_index += instance[p] * prev_card
                prev_card *= self.cardinalities[p]
            #print(v, instance[self.parents[v]], instance[v], par_index,
            #      self.cardinalities[self.parents[v]], self.parents[v])
        return int(par_index)

    def unveil_parent_joint_index(self, joint_index, v):
        if self.parents[v] is None:
            return None
        elif len(self.parents[v]) == 1:
            return np.array([joint_index])
        elif len(self.parents[v]) > 1:
            values = np.zeros(len(self.parents[v]))

            par_joint_card = np.prod(self.cardinalities[ self.parents[v][:-1] ])
            for j in np.flip(np.arange(1,len(self.parents[v]))):
                values[j] = joint_index // par_joint_card
                joint_index = joint_index % par_joint_card
                par_joint_card /= self.cardinalities[ self.parents[v][j-1] ]
            values[0] = joint_index

            return values.astype(int)
        else:
            print("I don't know how to deal with you, dude!")

    def bayes_error(self):
        vals = [np.arange(c) for c in self.cardinalities]
        vals[self.class_ind] = np.array([0])
        err = 0.0
        ind = 0.0
        for instance in product(*vals):
            probs = self.probability_distribution(np.asarray(instance))
            err += 1.0-np.max(probs)
            ind += 1
        return err/ind



class NaiveBayes(BNC):
    def __init__(self, class_ind, cardinalities, ordering=None):
        BNC.__init__(self, class_ind, cardinalities, ordering)

        for v in np.arange(len(self.cardinalities)):
            if v != self.class_ind:
                self.parents[v] = [self.class_ind]
                self.prob_distr[v] = np.zeros((self.cardinalities[v],
                                               self.cardinalities[self.class_ind]))
        self.prob_distr[self.class_ind] = np.zeros((self.cardinalities[self.class_ind],1))

    def learn(self, dataset, weights=None, smoothing=1):
        self.clear()
        if weights is None:
            weights = obtain_full_weights(dataset[:, self.class_ind],
                                          self.cardinalities[self.class_ind])

        for n in np.arange(dataset.shape[0]):
            self.prob_distr[self.class_ind][:, 0] += weights[n, :]

            for j in np.arange(len(self.cardinalities)):
                if j != self.class_ind:
                    self.prob_distr[j][dataset[n, j], :] += weights[n, :]

        self.prob_distr[self.class_ind] += smoothing  # Laplace smoothing
        self.prob_distr[self.class_ind] /= np.sum(self.prob_distr[self.class_ind])
        for j in np.arange(len(self.cardinalities)):
            if j != self.class_ind:
                self.prob_distr[j] += smoothing  # Laplace smoothing
                self.prob_distr[j] /= np.sum(self.prob_distr[j], 0)

    def probability_distribution(self, instance):
        probs = self.prob_distr[self.class_ind][:,0].copy()
        for v in np.arange(len(self.cardinalities)):
            if v != self.class_ind:
                probs *= self.prob_distr[v][instance[v],:]

        return probs/np.sum(probs)

    def random_generation(self, alpha_0=1):
        # the class variable has no parents
        # parents[0] = []
        self.prob_distr[self.class_ind] = np.random.dirichlet([alpha_0]*self.cardinalities[self.class_ind], 1).T

        for i in np.arange(1,len(self.cardinalities)):
            par_joint_card = self.cardinalities[self.class_ind]

            cond_prob_distrs = np.random.dirichlet([alpha_0] * self.cardinalities[self.ordering[i]],
                                                   par_joint_card).T

            self.parents[self.ordering[i]] = [self.class_ind]
            self.prob_distr[self.ordering[i]] = cond_prob_distrs


class KDB(BNC):
    def __init__(self, class_ind, cardinalities, ordering=None, K=2):
        BNC.__init__(self, class_ind, cardinalities, ordering)
        self.K = K

    def learn(self, dataset, weights=None, smoothing=1):
        self.clear()
        if weights is None:
            weights = obtain_full_weights(dataset[:,self.class_ind],
                                          self.cardinalities[self.class_ind])

        for n in np.arange(dataset.shape[0]):
            self.prob_distr[self.class_ind][:,0] += weights[n, :]

            weighted_class_values = np.where(weights[n, :] > 0)[0]
            for class_value in weighted_class_values:
                aux_instance = dataset[n, :].copy()
                aux_instance[self.class_ind] = class_value
                for v in self.ordering:
                    if v != self.class_ind:
                        self.prob_distr[v][aux_instance[v], self.parent_joint_index(aux_instance,v)] += weights[n, class_value]

        self.prob_distr[self.class_ind] += smoothing  # Laplace smoothing
        self.prob_distr[self.class_ind] /= np.sum(self.prob_distr[self.class_ind])
        for v in self.ordering:
            if v != self.class_ind:
                self.prob_distr[v] += smoothing  # Laplace smoothing
                self.prob_distr[v] /= np.sum(self.prob_distr[v], 0)

    # Sahami algorithm for structural learning
    def structural_learning(self, dataset):
        Iic, Iijc=mutual_informations(dataset)
        S=[]
        for i in np.arange(1,len(self.cardinalities)):
            xmax = np.argmax(Iic)
            self.ordering[i] = xmax

            m = min(len(S), self.K)
            ord_descvars = np.argsort(Iijc[xmax,:])[::-1]
            cand_parents = ord_descvars[:m]
            cand_parents = np.append(cand_parents, self.class_ind) # the class var is always a parent
            self.parents[self.ordering[i]] = cand_parents
            Iic[xmax] = -np.inf # avoid selecting xmax again
            S.append(xmax)

        #initialize parameters (empty)
        self.prob_distr[self.class_ind] = np.zeros((self.cardinalities[self.class_ind],1))
        for i in np.arange(1,len(self.cardinalities)):
            par_joint_card = np.prod(self.cardinalities[ self.parents[self.ordering[i]] ])
            self.prob_distr[self.ordering[i]] = np.zeros((self.cardinalities[self.ordering[i]], par_joint_card))

    def random_generation(self, alpha_0=1):
        # the class variable has no parents
        # parents[0] = []
        self.prob_distr[self.class_ind] = np.random.dirichlet([alpha_0]*self.cardinalities[self.class_ind], 1).T

        for i in np.arange(1,len(self.cardinalities)):
            cand_parents = self.ordering[1:i]
            if len(cand_parents) > self.K:
                cand_parents = np.random.choice(cand_parents, self.K, False)
            cand_parents = np.append(cand_parents, self.class_ind) # the class var is always a parent
            cand_parents = np.sort(cand_parents)
            par_joint_card = np.prod(self.cardinalities[cand_parents])

            cond_prob_distrs = np.random.dirichlet([alpha_0] * self.cardinalities[self.ordering[i]],
                                                   par_joint_card).T

            self.parents[self.ordering[i]] = cand_parents
            self.prob_distr[self.ordering[i]] = cond_prob_distrs
