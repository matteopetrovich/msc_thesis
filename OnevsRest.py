import numpy as np
import warnings
from sks3vm import SK_S3VM
from sklearn.preprocessing import LabelBinarizer


class ThreeClass_S3VM:
    
    def __init__(self, kernel = 'Linear', lam = 1.0, sigma = 0.5, lamU = 1.0):
        self.kernel = kernel
        self.lam = lam
        self.sigma = sigma
        self.lamU = lamU
        
        
    def fit_predict(self, X, y, X_test):
        
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        X_train = X
        Y_train = y
        
        X_test = X_test
        
        lab_bin = LabelBinarizer()

        y = lab_bin.fit_transform(Y_train)
        classes = lab_bin.classes_
        unlab_idx = np.where(classes==-1)[0]  #check which column has unlabelled coded as 1

        idx = np.where(y[:,unlab_idx] ==1)  # check the rows where unlabelled are 1
        
        y[idx[0], :]=-1  # change unlabelled documents in other columns to -1
        y = np.delete(y, idx, 1)
        y = y.astype('int')
        
        
        model = SK_S3VM( lam=self.lam, kernel=self.kernel, sigma=self.sigma, lamU=self.lamU)
        
        
        yhat_train = [] 
        yhat_f = []

        
        indicator_train= np.zeros((X_train.shape[0], 3))
        indicator_f = np.zeros((X_test.shape[0], 3))
        
        for i in range(3):
            Y = np.array(y[:,i]).flatten()
            fit = model.fit(X_train, Y)
            yhat_train.append(fit)
            yhat_f.append(model.predict_score(X_test))

        for i in range(X_train.shape[0]):
            arr = np.array([ yhat_train[0][i], yhat_train[1][i], yhat_train[2][i] ])    
            argm = np.argmax(arr) 
            indicator_train[i,argm] = 1
            
        for i in range(X_test.shape[0]):
            arr = np.array([ yhat_f[0][i], yhat_f[1][i], yhat_f[2][i] ])    
            argm = np.argmax(arr) 
            indicator_f[i,argm] = 1
        
  
    
        #creating column of zero for unlab to retrieve classes
        indicator_train = np.insert(indicator_train, unlab_idx, 0 , axis=1) 
        indicator_f = np.insert(indicator_f, unlab_idx, 0 , axis=1) 
        
        
        train_pred = lab_bin.inverse_transform(indicator_train)
        #retrieve classes

        pred = lab_bin.inverse_transform(indicator_f)
            
        d = dict()
        d['Train Predictions'] = train_pred
        d['Test Predictions'] = pred
        
        return d
        
    

        

        






