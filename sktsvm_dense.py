from sklearn.base import BaseEstimator
import random as rnd
import numpy
from qns3vm_p3 import QN_S3VM

class SK_S3VM(BaseEstimator):
    
    
    # lamU -- cost parameter that determines influence of unlabeled patterns (default 1, must be float > 0)
    def __init__(self, kernel = 'RBF', lam = 1e-4, sigma = 0.5, lamU = 1.0, score=True, probability=True):
        self.random_generator = rnd.Random()
        self.kernel = kernel
        self.lam = lam
        self.sigma = sigma 
        self.lamU = lamU
        self.score = score
        self.probability = probability
        
    def fit(self, X, y): # -1 for unlabeled
        
        # http://www.fabiangieseke.de/index.php/code/qns3vm
        
        
        unlabeledX = X[y==-1, :].tolist()
        labeledX = X[y!=-1, :].tolist()
        labeledy = y[y!=-1]
        # convert class 0 to -1 for tsvm
        labeledy[labeledy==0] = -1
            
        
            
        
        if 'rbf' in self.kernel.lower():
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.lam, lamU=self.lamU, kernel_type="RBF", sigma=self.sigma)
        else:
            self.model = QN_S3VM(labeledX, labeledy, unlabeledX, self.random_generator, lam=self.laam, lamU=self.lamU)
            
        train_pred = self.model.train()
        
        
            
        return train_pred
        

    def predict_score(self, X):
        """Compute scoring function wX + b for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        Y : array-like, shape = [n_samples, n_classes]
            Returns the scoring functions of the sample for each class in
            the model. 
        """
        if self.score:
            preds = self.model.predictValue(X)
            return preds
        

    def predict(self, X):
        """Compute class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        Y : array-like, shape = [n_samples, n_classes]
            Returns the class labels of the sample for each class in
            the model. 
        """
        
        y = numpy.array(self.model.predict(X.tolist()))
        y[y == -1] = 0
        return y
    
