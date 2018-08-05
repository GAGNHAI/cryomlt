import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from datetime import datetime
import gc
import learningLibMod as ll
import learningResults as res
import ml_RunDefinitions as rdefs

# Enable same process to run showing swath and Arctic DEM diffs
class Swath(object):
    def __init__(self,x_train):
        self._x_train = x_train
    def predict(self,x_test):
        return x_test.Elev_Swath*0
    def save(self,saveFolder):
        return 0
    @property
    def TimeTaken(self):
        return 0.0
    
class ArcticDEM(object):
    def __init__(self,x_train):
        self._x_train = x_train
    def predict(self,x_test):
        return -x_test.DemDiff_Swath
    def save(self,saveFolder):
        return 0
    @property
    def TimeTaken(self):
        return 0.0

def runLR(x_train,y_train,dropCols,scaleY):
    method = LinearRegression()
    model = ll.RegressionScikit()
    model.train(x_train,y_train,method,dropCols,scaleY)
    return model

def runNN(x_train,y_train,dropCols,lr,maxIter,lossFn,optimizer,scaleY):
    model = ll.RegressionTorch()
    model.train(x_train,y_train,dropCols,maxIter,lossFn=lossFn,learningRate = lr,optimizer=optimizer,scaleY=scaleY)
    return model

def runSVR(x_train,y_train,dropCols,scaleY,maxIter):
    method = SVR(kernel='rbf',epsilon=1e-3,C=1e2,max_iter=maxIter)
    #method = SVR(kernel='rbf',epsilon=1e-3,C=1e2,tol=1e-2)
    #method = SVR(kernel='rbf',epsilon=1e-3,C=4,tol=1e-2,max_iter=maxIter)
    model = ll.RegressionScikit()
    model.train(x_train,y_train,method,dropCols,scaleY)
    return model

def loadScikit(loadFolder):
    model = ll.RegressionScikit()
    model.load(loadFolder)
    return model

def loadNN(loadFolder):
    model = ll.RegressionTorch()
    model.load(loadFolder)
    return model

def runModel(modType,x_train,y_train,dropCols,saveFolder,lr=None,maxIter=None,lossFn=None,optimizer=None,scaleY=None):
    print("Starting at {}".format(time.ctime()))
    print ("Train Data Size: {}".format(x_train.shape[0]))
    if modType == 'LR':
        model = runLR(x_train,y_train,dropCols,scaleY)
    elif modType == 'SVR':
        model = runSVR(x_train,y_train,dropCols,scaleY,maxIter)
    elif modType == "NN":
        model = runNN(x_train,y_train,dropCols,lr,maxIter,lossFn,optimizer,scaleY)
    elif modType == "Swath":
        model = Swath(x_train)
    elif modType == "DEM":
        model = ArcticDEM(x_train)
    model.save(saveFolder)   
    print("Done at {}".format(time.ctime())) 
    return model

def loadModel(modType,loadFolder):
    print("Loading start at {}".format(time.ctime()))
    if modType == 'LR':
        model = loadScikit(loadFolder)
    elif modType == 'SVR':
        model = loadScikit(loadFolder)
    elif modType == "NN":
        model = loadNN(loadFolder)
    print("Done at {}".format(time.ctime())) 
    return model
