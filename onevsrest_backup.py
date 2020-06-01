
import os
os.chdir('C:/Users/petro/Desktop/PYTHON_THESIS')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sktsvm import SKTSVM
#from scipy import sparse

import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

filenames_train = ['cv_train.csv', 'tfidf_train.csv', 'w2v_train.csv']
train = pd.read_csv(filenames_train[2])
X_train = np.array(train.drop('SENTIMENT', axis=1))
#X_train= sparse.csc_matrix(X_train) # only if cv or tfidf
Y_train = np.array(train.SENTIMENT.values)
#Y_train = Y_train.astype('int')

filenames_test = ['cv_test.csv', 'tfidf_test.csv', 'w2v_test.csv']
test = pd.read_csv(filenames_test[2])
X_test = np.array(test.drop('SENTIMENT', axis=1))
#X_test= sparse.csc_matrix(X_test)
Y_test = np.array(test.SENTIMENT.values)
Y_test = Y_test.astype('int')


'''
SCORE FUNCTION

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score



def scores(Y_test, yhat ):
    acc = round(accuracy_score(Y_test, yhat) , 4)
    f = round(f1_score(Y_test, yhat, pos_label=1, average="weighted") , 4)
    rec = round(recall_score(Y_test, yhat, pos_label=1, average="weighted") , 4)
    prec = round(precision_score(Y_test, yhat, pos_label=1, average="weighted") , 4)
    scoring_temp = np.array([[acc, f, rec, prec]])
    scoring= pd.DataFrame(scoring_temp, columns=['Accuracy', 'F-Score', 'Recall', 'Precision'])
    return scoring

'''



'''
train labels for the 1 vs rest classifiers




lab_bin = LabelBinarizer()

y = lab_bin.fit_transform(Y_train)
classes = lab_bin.classes_
unlab_idx = np.where(classes==-1)[0] 

indx = np.where(y[:,unlab_idx] ==1)
y[indx[0], :]=-1
y = np.delete(y, unlab_idx, axis=1)
y = y.astype('int')
'''


'''
Obtaining predictions for each classifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
yhat_prob = []
for i in range(3):
    Y = np.array(y[:,i]).flatten()
    fit = semi_SVM.fit(X_train, Y)
    print  semi_SVM.predict_proba(X_test)
    print  semi_SVM.predict_score(X_test)
    yhat_prob.append(semi_SVM.predict_proba(X_test))

indicator = np.zeros((len(Y_test), 4))  
for i in range(len(Y_test)):
    arr = np.array([ yhat_prob[0][i][0], yhat_prob[1][i][0], yhat_prob[2][i][0] ])    
    argm = np.argmax(arr) + 1
    indicator[i,argm] = 1
    

yhat = lab_bin.inverse_transform(indicator)
'''






warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



lab_bin = LabelBinarizer()

y = lab_bin.fit_transform(Y_train)
classes = lab_bin.classes_
unlab_idx = np.where(classes==-1)[0]  #check which column has unlabelled coded as 1

idx = np.where(y[:,unlab_idx] ==1)  # check the rows where unlabelled are 1

y[idx[0], :]=-1  # change unlabelled documents in other columns to -1
y = np.delete(y, idx, 1)
y = y.astype('int')



model = SKTSVM(kernel = 'Linear',C=1000, lamU=0.00001)

yhat_f = []
yhat_prob = []
yhat_unlab = []

for i in range(3):
    Y = np.array(y[:,i]).flatten()
    fit = model.fit(X_train, Y)
    yhat_unlab.append(fit)
    yhat_f.append(model.predict_score(X_test))
    yhat_prob.append(model.predict_prob(X_test))
    
indicator_unlab = np.zeros((X_train.shape[0], 3))  
for i in range(X_train.shape[0]):
    arr = np.array([ yhat_unlab[0][i], yhat_unlab[1][i], yhat_unlab[2][i] ])    
    argm = np.argmax(arr) 
    indicator_unlab[i,argm] = 1

indicator = np.zeros((X_test.shape[0], 3))  
for i in range(X_test.shape[0]):
    arr = np.array([ yhat_f[0][i], yhat_f[1][i], yhat_f[2][i] ])    
    argm = np.argmax(arr) 
    indicator[i,argm] = 1
    
indicator_p = np.zeros((X_test.shape[0], 3))  
for i in range(X_test.shape[0]):
    arr = np.array([ yhat_prob[0][i][0], yhat_prob[1][i][0], yhat_prob[2][i][0] ])    
    argm = np.argmax(arr) 
    indicator_p[i,argm] = 1
    
#creating column of zero for unlab to retrieve classes  
indicator_train = np.insert(indicator_unlab, unlab_idx, 0 , axis=1) 
indicator = np.insert(indicator, unlab_idx, 0 , axis=1) 
indicator_p = np.insert(indicator, unlab_idx, 0 , axis=1) 
#retrieve classes
yhat = lab_bin.inverse_transform(indicator)
yhat_p = lab_bin.inverse_transform(indicator_p)
yhat_train = lab_bin.inverse_transform(indicator_train)

print(pd.crosstab(Y_test, yhat, rownames=['True'], colnames=['Predicted'], margins=True))
print(pd.crosstab(Y_test, yhat_p, rownames=['True'], colnames=['Predicted p'], margins=True))

print(pd.crosstab(Y_train, yhat_train, rownames=['True'], colnames=['Predicted p'], margins=True))



