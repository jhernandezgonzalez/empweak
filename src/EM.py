import numpy as np
import copy

class EM:
    def __init__(self, model, iClass, cardinalities, smoothing=1, verbose=False, epsilon=0.001, num_max_iterations=200):
        self.iClass = iClass
        self.cardinalities = cardinalities.copy()
        self.model = model
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.num_max_iterations = num_max_iterations
        self.verbose = verbose

    def run(self, dataset, cand_sets):
        self.dataset = dataset
        self.cand_sets = cand_sets

        self.initialize()

        convergence = False
        it = 0
        while not convergence:
            it += 1
            if self.verbose: print('Iteracion', it)
            # E-step
            self.EStep()

            # M-step
            self.MStep()

            # Test of convergence
            convergence = self.convergenceTest(it)

        return self.model, self.weights

    def initialize(self):
        self.initial_weights = np.zeros((len(self.cand_sets), self.cardinalities[self.iClass]))
        for n in np.arange(self.initial_weights.shape[0]):
            self.initial_weights[n,self.cand_sets[n]] = 1
        self.initial_weights /= np.sum(self.initial_weights, 1, keepdims=True)#self.cardinalities[self.iClass]

        self.model.clear()
        self.model.learn(self.dataset, self.initial_weights, self.smoothing)

    def EStep(self):
        self.weights = np.zeros(self.initial_weights.shape)
        for n in np.arange(self.dataset.shape[0]):
            self.weights[n, :] = self.model.probability_distribution(self.dataset[n, :])
        self.weights[np.where(self.initial_weights == 0)] = 0
        self.weights /= np.sum(self.weights, 1, keepdims=True)#self.cardinalities[self.iClass]

    def MStep(self):
        self.prev_model = copy.deepcopy(self.model)

        self.model.clear()
        self.model.learn(self.dataset, self.weights, self.smoothing)

    def convergenceTest(self, num_its):
        diff = self.model.compare_to(self.prev_model)

        return diff < self.epsilon or num_its >= self.num_max_iterations
