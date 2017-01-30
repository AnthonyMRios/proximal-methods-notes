from time import time
import sys
import numpy as np

class GSML:
    def __init__(self, max_iters=8, gamma=10, learning_rate=0.01, margin=1.):
        self.max_iters = max_iters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.margin = margin

    def _dhinge(self, x1, x2, x3, margin=1.):
        score = margin + (x1-x2).dot(self.W).dot(x1-x2) -\
                (x1-x3).dot(self.W).dot(x1-x3)
        if score > 0:
            return 1.
        else:
            return 0.

    def fit(self, X, triplets, L=None):
        if L is None:
            self.L = np.eye(X.shape[1])

        self.W = np.eye(X.shape[1])
        active_set = set()
        working_set = set()
        converged = False
        cur_iter = 0
        Nt = set()
        Gt = self.gamma*self.L
        while cur_iter < self.max_iters:
            print cur_iter
            sys.stdout.flush()
            t0 = time()
            Gtp1 = Gt
            Ntp1 = set()
            if cur_iter == 0:
                for index in np.arange(triplets.shape[0]):
                    x1 = X[triplets[index][0]]
                    x2 = X[triplets[index][1]]
                    x3 = X[triplets[index][2]]
                    if self._dhinge(x1, x2, x3, self.margin) > 0:
                        Ntp1.add(index)
                        Cij = np.outer(x1-x2, x1-x2)
                        Cik = np.outer(x1-x3, x1-x3)
                        Gtp1 += Cij - Cik
            else:
                x1 = X[triplets[:,0]]
                x2 = X[triplets[:,1]]
                x3 = X[triplets[:,2]]
                x12 = (x1-x2).dot(self.W).dot((x1-x2).T)
                x13 = (x1-x3).dot(self.W).dot((x1-x3).T)
                final = self.margin+x12.diagonal()-x13.diagonal()
                Ntp1 = set(np.where(final.flatten() > 0)[0].tolist())
                if len(Ntp1) == 0:
                    converged = True
                    break
                for index in Nt - Ntp1:
                    x1 = X[triplets[index][0]]
                    x2 = X[triplets[index][1]]
                    x3 = X[triplets[index][2]]
                    Cij = np.outer(x1-x2, x1-x2)
                    Cik = np.outer(x1-x3, x1-x3)
                    Gtp1 -= Cij - Cik
                for index in Ntp1 - Nt:
                    x1 = X[triplets[index][0]]
                    x2 = X[triplets[index][1]]
                    x3 = X[triplets[index][2]]
                    Cij = np.outer(x1-x2, x1-x2)
                    Cik = np.outer(x1-x3, x1-x3)
                    Gtp1 += Cij - Cik

            self.W -= self.learning_rate*Gtp1
            eigval, eigvec = np.linalg.eig(self.W)
            diag_max_eigval = np.diag(np.maximum(eigval, 0))
            self.W = eigvec.dot(diag_max_eigval).dot(eigvec.T)

            #Nt = Nt.union(Ntp1)
            print cur_iter, 'done:', time()-t0, len(Ntp1), len(Ntp1-Nt), len(Nt-Ntp1)
            sys.stdout.flush()
            Nt = Ntp1
            Gt = Gtp1
            cur_iter += 1
            self.learning_rate = self.learning_rate/2.

    def distance(self, x1, x2):
        return (x1-x2).dot(self.W).dot(x1-x2)
