#import tensorflow as tf
import numpy as np
import numpy
import scipy.io as sio
import sklearn
from sklearn import svm as sksvm
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.model_selection import GridSearchCV
#import sklearn
#import matplotlib.pyplot as plt
import h5py
from dataClass import *
NORMALISE = True;
LOGNORM=False;



batch_size = 200;
bSize = 200;
nfeat = 46;

def bilog(x):
    return np.multiply(np.sign(x),np.log(np.abs(x)+1))
def log(x):
    return np.log(x)


def quantify(pred,lab):

    lab = 1-lab;
    pred = 1-pred;

    TP = np.sum(np.multiply(lab,pred));
    FP = np.sum(np.multiply(1-lab,pred));
    FN = np.sum(np.multiply(lab,1-pred));
    TN = np.sum(np.multiply(1-lab,1-pred));
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = (2*(prec*rec))/(prec+rec)
    acc = 1- (np.sum(np.abs(lab-pred))/np.shape(lab)[1])
    print('acc = {}'.format(acc))
    print('prec = {}'.format(prec))
    print('rec = {}'.format(rec))
    print('f1 = {}'.format(f1))

    sens = TP/(TP+FN);
    spec = TN/(TN+FP);
    print('sens = {}'.format(sens))
    print('spec = {}'.format(spec))

    return acc,prec,rec,f1

def combData(x, test=1):
    if test == 1:

        return np.concatenate((x.valid.features, x.test.features), axis=0),np.concatenate((x.valid.labels, x.test.labels), axis=0)

    else:
        #np.concatenate((x.train.features,x.valid.features,x.test.features),axis=0)
        return np.concatenate((x.train.features,x.valid.features,x.test.features),axis=0),np.concatenate((x.train.labels,x.valid.labels,x.test.labels),axis=0)
# load dataset
loadname = '/cs/research/medim/gdisciac/SOLUS/example/example_202110Wrapper/Results/SOLUSdata4classification.mat'
#loadname = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/CLASSIFICATION_538'

#dataset = fullDataset(loadname);

fData = h5py.File(loadname, 'r')
features = numpy.array(fData.get('features'))
labels = numpy.array(fData.get('labels'))

if LOGNORM==True:
    minin = 1+np.min(features)
    features = np.log(minin+features)

#print(dataset.train.features[0,:])
if NORMALISE==True:
    concd = features;
    nmean = np.mean(concd, axis=0)
    nstd= np.std( concd, axis=0)
    features = np.divide(features - nmean,nstd)
#if PCA == True:
#    sklearn.decomposition.PCA

svmpred = np.zeros(np.shape(labels)[0]);
lgrpred = np.zeros(np.shape(labels)[0]);

for i_out in range(0, np.shape(labels)[0]):
    print(i_out)
    v = np.ones(features.shape[1])==1;
    v[i_out] = False;
    X = np.array(features[:,v].T);
    Y = np.array(labels[v], dtype=np.int32)

    #parameters = {'kernel': ['rbf', 'poly', 'sigmoid'],'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 50, 75, 100], 'max_iter': [6000]}

    parameters = {'kernel': ['rbf', 'poly', 'sigmoid'],
                  'C': [10000], 'max_iter': [6000]}

    clf = GridSearchCV(estimator=sksvm.SVC(), param_grid=parameters)
    clf.fit(X, np.ravel(Y))

    parameters = {'C': [ 10000],'max_iter': [6000]}#parameters = {'C': [ 1000],'max_iter': [6000]}
    lgr = GridSearchCV(estimator=logReg(), param_grid=parameters)
    lgr.fit(X, np.ravel(Y))

    svmpred[i_out] = clf.predict(np.array(features[:,v==False].T))
    lgrpred[i_out] = lgr.predict(np.array(features[:,v==False].T))



print('svm')
quantify(svmpred,labels.T)
print('lgrpred')
quantify(lgrpred,labels.T)


svname = '/cs/research/medim/gdisciac/SOLUS/example/example_202110Wrapper/Results/Predictions.mat'

svdict={'svmpred':svmpred,'lgrpred':lgrpred}

sio.savemat(svname, svdict)



