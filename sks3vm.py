from sklearn.base import BaseEstimator
import random as rnd
import numpy
from qns3vm_p3 import QN_S3VM
import scipy.sparse.csc as csc


class SK_S3VM(BaseEstimator):
 

    def __init__(self, kernel = 'RBF', lam = 1, sigma = 0.5, lamU = 1.0, score=True):
        self.random_generator = rnd.Random()
        self.kernel = kernel
        self.lam = lam
        self.sigma = sigma
        self.lamU = lamU
        self.score = score
       
        
        
        
    def fit(self, X, y): # -1 for unlabeled

        
        if isinstance(X, csc.csc_matrix):
            unlabeledX = X[y==-1, :]
            labeledX = X[y!=-1, :]
        else:
            unlabeledX = X[y==-1, :].tolist()
            labeledX = X[y!=-1, :].tolist()
        
        labeledy = y[y!=-1]
        # convert class 0 to -1 for tsvm
        labeledy[labeledy==0] = -1
            
        
        if 'rbf' in self.kernel.lower():
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.lam, lamU=self.lamU, kernel_type="RBF", sigma=self.sigma)
        else:
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.lam, lamU=self.lamU)
            
        train_pred = self.model.train()
        
        
        return train_pred
        




    def predict_score(self, X):
 
        if self.score:
            preds = self.model.predictValue(X)
            return preds



        

    def predict(self, X):

        
        y = numpy.array(self.model.predict(X.tolist()))
        y[y == -1] = 0
        return y
    

    