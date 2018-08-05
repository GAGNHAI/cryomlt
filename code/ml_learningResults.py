#!/usr/bin/env python2

''' ML results analysis '''

#Libaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
import os


class ResultCollection(object):
    ''' A collection of results for multiple train/test sets '''
    def __init__(self):
        ''' Initialise and define result metrics/structure'''
        self._df = pd.DataFrame(columns=['TrainSet','TestSet','MeanDiff','MedDiff','Count<1m','CI(z=1)','StdDev','RMSE','Mad','MinDiff','MaxDiff','N','Time(mins)'])
        self._oibSummary = pd.DataFrame(columns=['Set','NPoints','Mean','CI(z=1)','Std','Min','Max'])
        self._hists = {}
        self._density = {}
        
    def add(self,trainName,testName,resultSet,timeTaken):
        ''' Add train/test set result'''
        n = resultSet.shape[0]
        self._addTable(trainName,testName,resultSet['Difference'],n,timeTaken)       
        gc.collect()
        
    def _addTable(self,trainName,testName,diffs,n,timeTaken):
        ''' Calculate metrics to add to table'''
        rMSE = np.sqrt(np.mean((diffs)**2))
        meanDiff = np.mean(diffs)
        medDiff = np.median(diffs)
        minDiff = np.min(diffs)
        maxDiff = np.max(diffs)
        std = np.std(diffs)
        conf = 1*std/np.sqrt(n)
        count1m = np.where(np.abs(diffs)<1)[0].shape[0]
        mad = np.median(np.abs(diffs - np.median(diffs)))
        timeMins = timeTaken/60.0
        self._df = self._df.append({'TrainSet':trainName,'TestSet':testName,'MeanDiff':meanDiff,'MedDiff':medDiff,'CI(z=1)':conf,'Count<1m':count1m,'Mad':mad,'MinDiff':minDiff,'MaxDiff':maxDiff,'StdDev':std,'RMSE':rMSE,'N':n,'Time(mins)':timeMins}, ignore_index=True)
   
    def __repr__(self):
        ''' Return self '''
        return repr(self._df)
    
    def _addHist(self,scenario,diffs):
        ''' Plot histogram '''
        fig = plt.figure()
        plt.hist(diffs,bins=101,range=[-500,500], figure=fig)
        plt.ylabel('Count', figure=fig)
        plt.xlabel('Predicted - OIB (m)', figure=fig)
        plt.title(scenario, figure=fig)
        self._hists[scenario] = fig
    
    def save(self,path,fileName):
        ''' Save results CSV '''
        if not os.path.exists(path):
            os.makedirs(path)
        self._df.to_csv(path+"/"+fileName+".csv")
