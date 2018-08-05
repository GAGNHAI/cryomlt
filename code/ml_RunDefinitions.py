#!/usr/bin/env python2

'''
ML Run definitions
This contains combinations of lists of training/test sets for use in the ML code
'''

import numpy as np
from collections import OrderedDict

class RunSet(object):
    ''' A runset object containing train/test mappings '''
    
    def __init__(self,trainSet,testSets,store):          
        self._trainSet = trainSet
        self._testSets = testSets
        self._store = store

    def getSet(self,setList):
        ''' Returns a set of data '''
        setNames = setList[0]
        data = self._store[setNames]
        for i in np.arange(1,len(setList)):
            setName = setList[i]
            setNames += setName
            data = data.append(self._store[setName])
        return setNames, data  
      
    def getTrainSet(self):
        ''' Returns a training set '''
        setNames, data = self.getSet(self._trainSet)
        return setNames, data
    
    def getTestSets(self):
        ''' Returns the set of test sets '''
        setDict = OrderedDict()
        setLists = self._testSets
        for i in np.arange(0,len(setLists)):
            setName, data = self.getSet(setLists[i])
            setDict[setName] = data
        return setDict
    
    def getTestKeys(self):
        return self._testSets
    
def getRunList(store,name):
    '''
    Returns a runset for the different training/test set collections
    collection names are: big, medium, small, verySmall, single, scenario, plots, compare
    '''
    
    if name == 'big':
        return bigRunList(store)
    elif name == 'medium':
        return mediumRunList(store)
    elif name == 'small':
        return smallRunList(store)
    elif name == 'verySmall':
        return verySmallRunList(store)
    elif name == 'single':
        return singleRunList(store)
    elif name == 'scenario':
        return scenarioRunList(store)
    elif name == 'plots':
        return plotList(store)
    elif name == 'compare':
        return compareList(store)                 

def bigRunList(store):
    runDefs = OrderedDict()

    #Within
    runDefs['jak11train'] = RunSet(['jak11train'],[['jak11test']],store)
    runDefs['st11train'] = RunSet(['st11train'],[['st11test']],store)
    runDefs['se11train'] = RunSet(['se11train'],[['se11test']],store)
    runDefs['all11train'] = RunSet(['jak11train','st11train','se11train'],[['jak11test','st11test','se11test']],store)
    runDefs['jak14train'] = RunSet(['jak14train'],[['jak14test']],store)
    runDefs['st14train'] = RunSet(['st14train'],[['st14test']],store)
    runDefs['se14train'] = RunSet(['se14train'],[['se14test']],store)
    runDefs['all14train'] = RunSet(['jak14train','st14train','se14train'],[['jak14test','st14test','se14test']],store)
    runDefs['all11to15train'] = RunSet(['jak11train','st11train','se11train','jak12train','st12train','se12train','jak13train','st13train','se13train','jak14train','st14train','se14train','jak15train','st15train','se15train'],[['jak11test','st11test','se11test','jak12test','st12test','se12test','jak13test','st13test','se13test','jak14test','st14test','se14test','jak15test','st15test','se15test']],store)   
    runDefs['all11to16train'] = RunSet(['jak11train','st11train','se11train','jak12train','st12train','se12train','jak13train','st13train','se13train','jak14train','st14train','se14train','jak15train','st15train','se15train','jak16train','st16train','se16train'],[['jak11test','st11test','se11test','jak12test','st12test','se12test','jak13test','st13test','se13test','jak14test','st14test','se14test','jak15test','st15test','se15test','jak16test','st16test','se16test']],store)   

    #Single
    runDefs['jak11'] = RunSet(['jak11'],[['st11'],['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16']],store)
    runDefs['st11'] = RunSet(['st11'],[['jak11'],['se11'],['st12'],['st13'],['st14'],['st15'],['st16']],store)
    runDefs['se11'] = RunSet(['se11'],[['jak11'],['st11'],['se12'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['jak14'] = RunSet(['jak14'],[['st14'],['se14'],['jak15'],['jak16']],store)
    runDefs['st14'] = RunSet(['st14'],[['jak14'],['se14'],['st15'],['st16']],store)
    runDefs['se14'] = RunSet(['se14'],[['jak14'],['st14'],['se15'],['se16']],store)
    runDefs['jak11to14'] = RunSet(['jak11','jak12','jak13','jak14'],[['st11','st12','st13','st14'],['se11','se12','se13','se14'],['jak15'],['jak16']],store)
    runDefs['st11to14'] = RunSet(['st11','st12','st13','st14'],[['jak11','jak12','jak13','jak14'],['se11','se12','se13','se14'],['st15'],['st16']],store)
    runDefs['se11to14'] = RunSet(['se11','se12','se13','se14'],[['jak11','jak12','jak13','jak14'],['st11','st12','st13','st14'],['se15'],['se16']],store)
    runDefs['jak11to15'] = RunSet(['jak11','jak12','jak13','jak14','jak15'],[['st11','st12','st13','st14','st15'],['se11','se12','se13','se14','se15'],['jak16']],store)
    runDefs['st11to15'] = RunSet(['st11','st12','st13','st14','st15'],[['jak11','jak12','jak13','jak14','jak15'],['se11','se12','se13','se14','se15'],['st16']],store)
    runDefs['se11to15'] = RunSet(['se11','se12','se13','se14','se15'],[['jak11','jak12','jak13','jak14','jak15'],['st11','st12','st13','st14','st15'],['se16']],store)     

    #Double
    runDefs['jakst11'] = RunSet(['jak11','st11'],[['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16'],['st12'],['st13'],['st14'],['st15'],['st16']],store)
    runDefs['jakse11'] = RunSet(['jak11','se11'],[['st11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16'],['se12'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['stse11'] = RunSet(['st11','se11'],[['jak11'],['st12'],['st13'],['st14'],['st15'],['st16'],['se12'],['se13'],['se14'],['se15'],['se16']],store)    
    
    runDefs['jakst11to12'] = RunSet(['jak11','st11','jak12','st12'],[['se11'],['se12'],['jak13'],['jak14'],['jak15'],['jak16'],['st13'],['st14'],['st15'],['st16']],store)
    runDefs['jakse11to12'] = RunSet(['jak11','se11','jak12','se12'],[['st11'],['st12'],['jak13'],['jak14'],['jak15'],['jak16'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['stse11to12'] = RunSet(['st11','se11','st12','se12'],[['jak11'],['jak12'],['st13'],['st14'],['st15'],['st16'],['se13'],['se14'],['se15'],['se16']],store)     
  
    runDefs['jakst11to15'] = RunSet(['jak11','st11','jak12','st12','jak13','st13','jak14','st14','jak15','st15'],[['se11'],['se12'],['se13'],['se14'],['se15'],['jak16'],['st16']],store)
    runDefs['jakse11to15'] = RunSet(['jak11','se11','jak12','se12','jak13','se13','jak14','se14','jak15','se15'],[['st11'],['st12'],['st13'],['st14'],['st15'],['jak16'],['se16']],store)    
    runDefs['stse11to15'] = RunSet(['st11','se11','st12','se12','st13','se13','st14','se14','st15','se15'],[['jak11'],['jak12'],['jak13'],['jak14'],['jak15'],['st16'],['se16']],store)  
 
    # Temporal    
    runDefs['all11'] = RunSet(['jak11','st11','se11'],[['jak12','st12','se12'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16'],['st12'],['st13'],['st14'],['st15'],['st16'],['se12'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['all12'] = RunSet(['jak12','st12','se12'],[['jak13','st13','se13'],['jak13'],['jak14'],['jak15'],['jak16'],['st13'],['st14'],['st15'],['st16'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['all11to12'] = RunSet(['jak11','st11','se11','jak12','st12','se12'],[['jak13','st13','se13'],['jak13'],['jak14'],['jak15'],['jak16'],['st13'],['st14'],['st15'],['st16'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak15','st15','se15'],['jak15'],['jak16'],['st15'],['st16'],['se15'],['se16']],store)
    runDefs['all11to15'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14','jak15','st15','se15'],[['jak16','st16','se16'],['jak16'],['st16'],['se16']],store)
    
    return runDefs

def mediumRunList(store):
    runDefs = OrderedDict()

    #Within
    runDefs['jak11train'] = RunSet(['jak11train'],[['jak11test']],store)
    runDefs['st11train'] = RunSet(['st11train'],[['st11test']],store)
    runDefs['se11train'] = RunSet(['se11train'],[['se11test']],store)
    runDefs['all11train'] = RunSet(['jak11train','st11train','se11train'],[['jak11test','st11test','se11test']],store)
    runDefs['jak14train'] = RunSet(['jak14train'],[['jak14test']],store)
    runDefs['st14train'] = RunSet(['st14train'],[['st14test']],store)
    runDefs['se14train'] = RunSet(['se14train'],[['se14test']],store)
    runDefs['all14train'] = RunSet(['jak14train','st14train','se14train'],[['jak14test','st14test','se14test']],store)
    runDefs['all11to15train'] = RunSet(['jak11train','st11train','se11train','jak12train','st12train','se12train','jak13train','st13train','se13train','jak14train','st14train','se14train','jak15train','st15train','se15train'],[['jak11test','st11test','se11test','jak12test','st12test','se12test','jak13test','st13test','se13test','jak14test','st14test','se14test','jak15test','st15test','se15test']],store)   
    runDefs['all11to16train'] = RunSet(['jak11train','st11train','se11train','jak12train','st12train','se12train','jak13train','st13train','se13train','jak14train','st14train','se14train','jak15train','st15train','se15train','jak16train','st16train','se16train'],[['jak11test','st11test','se11test','jak12test','st12test','se12test','jak13test','st13test','se13test','jak14test','st14test','se14test','jak15test','st15test','se15test','jak16test','st16test','se16test']],store)   

    #Single
    runDefs['jak11'] = RunSet(['jak11'],[['st11'],['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16']],store)
    runDefs['st11'] = RunSet(['st11'],[['jak11'],['se11'],['st12'],['st13'],['st14'],['st15'],['st16']],store)
    runDefs['se11'] = RunSet(['se11'],[['jak11'],['st11'],['se12'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['jak14'] = RunSet(['jak14'],[['st14'],['se14'],['jak15'],['jak16']],store)
    runDefs['st14'] = RunSet(['st14'],[['jak14'],['se14'],['st15'],['st16']],store)
    runDefs['se14'] = RunSet(['se14'],[['jak14'],['st14'],['se15'],['se16']],store)

    #Double
    runDefs['jakst11'] = RunSet(['jak11','st11'],[['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16'],['st12'],['st13'],['st14'],['st15'],['st16']],store)

    #Temporal    
    runDefs['all11to12'] = RunSet(['jak11','st11','se11','jak12','st12','se12'],[['jak13','st13','se13'],['jak13'],['jak14'],['jak15'],['jak16'],['st13'],['st14'],['st15'],['st16'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak15','st15','se15'],['jak15'],['jak16'],['st15'],['st16'],['se15'],['se16']],store)

    return runDefs

def smallRunList(store):
    runDefs = OrderedDict()

    #Within
    runDefs['jak11train'] = RunSet(['jak11train'],[['jak11test']],store)
    runDefs['st11train'] = RunSet(['st11train'],[['st11test']],store)
    runDefs['se11train'] = RunSet(['se11train'],[['se11test']],store)
    runDefs['all11train'] = RunSet(['jak11train','st11train','se11train'],[['jak11test','st11test','se11test']],store)
    runDefs['all11to16train'] = RunSet(['jak11train','st11train','se11train','jak12train','st12train','se12train','jak13train','st13train','se13train','jak14train','st14train','se14train','jak15train','st15train','se15train','jak16train','st16train','se16train'],[['jak11test','st11test','se11test','jak12test','st12test','se12test','jak13test','st13test','se13test','jak14test','st14test','se14test','jak15test','st15test','se15test','jak16test','st16test','se16test']],store)   

    #Single
    runDefs['jak11'] = RunSet(['jak11'],[['st11'],['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16']],store)
    runDefs['st11'] = RunSet(['st11'],[['jak11'],['se11'],['st12'],['st13'],['st14'],['st15'],['st16']],store)
    runDefs['se11'] = RunSet(['se11'],[['jak11'],['st11'],['se12'],['se13'],['se14'],['se15'],['se16']],store)

    #Double
    runDefs['jakst11'] = RunSet(['jak11','st11'],[['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16'],['st12'],['st13'],['st14'],['st15'],['st16']],store)

    #Temporal 
    runDefs['all11to12'] = RunSet(['jak11','st11','se11','jak12','st12','se12'],[['jak13','st13','se13'],['jak13'],['jak14'],['jak15'],['jak16'],['st13'],['st14'],['st15'],['st16'],['se13'],['se14'],['se15'],['se16']],store)
    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak15','st15','se15'],['jak15'],['jak16'],['st15'],['st16'],['se15'],['se16']],store)

    
    return runDefs

def verySmallRunList(store):
    runDefs = OrderedDict()

    #Within
    runDefs['jak11train'] = RunSet(['jak11train'],[['jak11test']],store)
    runDefs['all11to16train'] = RunSet(['jak11train','st11train','se11train','jak12train','st12train','se12train','jak13train','st13train','se13train','jak14train','st14train','se14train','jak15train','st15train','se15train','jak16train','st16train','se16train'],[['jak11test','st11test','se11test','jak12test','st12test','se12test','jak13test','st13test','se13test','jak14test','st14test','se14test','jak15test','st15test','se15test','jak16test','st16test','se16test']],store)   

    #Double
    runDefs['jakst11'] = RunSet(['jak11','st11'],[['se11'],['jak12'],['jak13'],['jak14'],['jak15'],['jak16'],['st12'],['st13'],['st14'],['st15'],['st16']],store)
   
    #Temporal 
    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak15','st15','se15'],['jak15'],['jak16'],['st15'],['st16'],['se15'],['se16']],store)

    return runDefs

def plotList(store):
    runDefs = OrderedDict()

    #Within
    runDefs['jak11train'] = RunSet(['jak11train'],[['jak11test']],store)
    runDefs['st11train'] = RunSet(['st11train'],[['st11test']],store)
    runDefs['se11train'] = RunSet(['se11train'],[['se11test']],store)

    #Temporal 
    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak15','st15','se15'],['jak15'],['jak16'],['st15'],['st16'],['se15'],['se16']],store)

    return runDefs

def singleRunList(store):
    runDefs = OrderedDict()

    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak15','st15','se15'],['jak15'],['jak16'],['st15'],['st16'],['se15'],['se16']],store)

    return runDefs

def compareList(store):
    runDefs = OrderedDict()

    #Within
    runDefs['jak11'] = RunSet(['jak11'],[['jak11']],store)
    runDefs['jak12'] = RunSet(['jak12'],[['jak12']],store)
    runDefs['st11'] = RunSet(['st11'],[['st11']],store)
    runDefs['st12'] = RunSet(['st12'],[['st12']],store)
    runDefs['se11'] = RunSet(['se11'],[['se11']],store)
    runDefs['se12'] = RunSet(['se12'],[['se12']],store)

    return runDefs


def scenarioRunList(store):
    runDefs = OrderedDict()

    #Temporal Only
    runDefs['all11to14'] = RunSet(['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],[['jak11','st11','se11','jak12','st12','se12','jak13','st13','se13','jak14','st14','se14'],['jak15','st15','se15'],['jak11'],['st11'],['se11'],['jak12'],['st12'],['se12'],['jak13'],['st13'],['se13'],['jak14'],['st14'],['se14'],['jak15'],['st15'],['se15'],['jak16'],['st16'],['se16']],store)

    return runDefs

