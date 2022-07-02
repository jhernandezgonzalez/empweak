import fnmatch
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import fisher_exact


def brier_score(real_labels, pred_probs):
    orig_probs = np.zeros(pred_probs.shape)
    for n, lab in enumerate(real_labels):
        orig_probs[n, lab] = 1

    score = np.sum((pred_probs-orig_probs)**2)/len(real_labels)

    return score


def obtain_full_weights(labels, class_card):
    weis = np.zeros((len(labels),class_card))

    for i in np.arange(weis.shape[0]):
        weis[i,labels[i]] = 1

    return weis

def find_files(base, pattern):
    """Return list of files matching pattern in base folder."""
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
            os.path.isfile(os.path.join(base, n))]


def mutual_informations(D):
    '''
    D: np.array
    La clase es la ultima variable

    Como lo hacemos?
    I(Xi;C)=H(Xi)+H(C)-H(Xi,C)
    I(Xi;Xj|C)= H(Xi,C)+H(Xj,C)-H(C)-H(Xi,Xj,C)

    return I(Xi,C) in an np.array of lenght d-1,
    and I(Xi,Xj|C) in an array of length (d-1,d-1) with -inf in the diagonal
    '''
    n,d= D.shape
    p= np.unique(D[:,-1],return_counts=True)[1]/n

    #H(C)
    Hc= np.sum(-p[c]* np.log2(p[c]) for c in range(len(p)))
    #H(Xi)
    Hi= np.zeros(d-1)
    #H(Xi,C)
    Hic= np.zeros(d-1)
    #H(Xi,Xj,C)
    Hijc= np.zeros((d-1,d-1))
    #I(Xi,C)
    Iic= np.zeros(d-1)
    # I(Xi,Xj|C)
    Iijc= np.zeros((d-1,d-1))

    ri= np.zeros(d-1)
    for i in range(d-1):
        (xi,p)= np.unique(D[:,-1],return_counts=True)
        ri[i]= len(xi)
        p=p/n
        Hi[i]= np.sum(-p[i]* np.log2(p[i]) for i in range(len(p)))
        (xc,p)= np.unique(D[:,-1]*ri[i]+D[:,i], return_counts= True)
        p= p/n
        Hic[i]= np.sum(-p[i]* np.log2(p[i]) for i in range(len(p)))
        Iic[i]= Hc+Hi[i]-Hic[i]

    for i in range(d-1):
        Iijc[i,i]= -np.inf
        for j in range(i+1,d-1):
            (xyc, p) = np.unique(D[:, -1] * ri[i]*ri[j] + D[:, i]*ri[j] + D[:,j], return_counts=True)
            p= p/n
            Hijc[i,j] = np.sum(-p[i] * np.log2(p[i]) for i in range(len(p)))
            Iijc[i,j] = Hic[i]+Hic[j]-Hc-Hijc[i,j]
            Iijc[j,i]= Iijc[i,j]

    return Iic, Iijc


def realDataLoader(filename):
    data = loadmat(filename)
    X, y, cand_sets = data['features'], data['logitlabels'], data['p_labels']
    if X.shape[0] != y.shape[0]:
        y = y.transpose()
        cand_sets = cand_sets.transpose()
    if type(y) != np.ndarray:
        y = y.toarray()
        cand_sets = cand_sets.toarray()
    return X, y, cand_sets

class PLDataset:
    def __init__(self, X, y, cand_sets, name=None):
        self.X = X
        self.y = y
        self.cand_sets = cand_sets
        if name is not None:
            self.name=name

    def subset(self, n_retain, avoid_first=0):
        class_distr = np.sum(self.y, axis=0)
        mpcd = np.argsort(class_distr)[::-1]
        mpcd = mpcd[avoid_first:(n_retain+avoid_first)]
        sel_instances = np.sum(self.y[:, mpcd], axis=1) == 1
        return PLDataset(self.X[sel_instances, :],
                         self.y[np.ix_(sel_instances, mpcd)],
                         self.cand_sets[np.ix_(sel_instances, mpcd)],
                         name=self.name + "_subset_" + str(n_retain))

    def binarize(self):
        for col in np.arange(self.X.shape[1]):
            self.X[:, col] = np.digitize(self.X[:, col], [np.quantile(self.X[:, col], 0.5)])

    def remove_no_info(self, epsilon=1e-3):
        self.X = self.X[:, self.X.std(axis=0) >= epsilon]

    def remove_redundant_test(self, alpha=0.01):
        i = 0
        while i < self.X.shape[1]:
            to_keep = list(np.arange(i+1))
            for j in np.arange(i+1,self.X.shape[1]):
                # contingency table
                table = contingency_table(self.X[:, i], self.X[:, j])
                #print(i ,j )
                #print(table)
                _, p = fisher_exact(table)
                # interpret p-value
                if p > alpha:
                    # print('Independent (fail to reject H0, P=',p,')')
                    to_keep.append(j)
                #else:
                    #  print('Dependent (reject H0, P=',p,')')

            self.X = self.X[:,to_keep]
            print(to_keep)
            i += 1

    def remove_redundant_corr(self, redundancy=0.9):
        df = pd.DataFrame(self.X)
        corr = df.corr()
        #print("Initial number of columns:",corr.shape[0])
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= redundancy:
                    if columns[j]:
                        columns[j] = False
        selected_columns = df.columns[columns]
        #print("Final number of columns:",selected_columns.shape)

        df = df[selected_columns]
        self.X = df.to_numpy()

    def return_dataset(self):
        cardinalities = list(self.X.max(axis=0)+1)
        cardinalities.append(self.y.shape[1])
        cardinalities = np.array(cardinalities).astype(int)
        iclass = len(cardinalities) - 1
        data = np.append(self.X,np.argmax(self.y,axis=1).reshape(self.X.shape[0],1),axis=1)
        candsets = [ np.where(self.cand_sets[i,:]>0)[0] for i in np.arange(self.X.shape[0])]
        return data.astype(int), candsets, iclass, cardinalities

    def return_dataset_onlyfull(self):
        data, candsets, iclass, cardinalities = self.return_dataset()
        to_remain = np.where([len(candsets[i])==1 for i in np.arange(data.shape[0])])[0]
        tr_candsets = [np.array(candsets[i]) for i in to_remain]
        return data[to_remain,:], tr_candsets, iclass, cardinalities

def contingency_table(A, B):
    K = len(np.unique(A))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(A)):
        result[int(A[i]),int(B[i])] += 1

    return result


def indx_sample_stratified(points, ref_column, N):
    arr = np.column_stack((points,ref_column))
    df = pd.DataFrame(data=arr, columns=["points", "ref"])
    # perform stratified random sampling
    df_strat = df.groupby('ref', group_keys=False).apply(lambda x: x.sample(int(np.rint(N * len(x) / len(df))))).sample(
        frac=1).reset_index(drop=True)
    return np.sort(df_strat["points"].to_numpy())
