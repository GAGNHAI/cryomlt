#!/usr/bin/env python2

'''
Loads the saved ML models and applies to other data

Instructions:
    1) Set configuration section below
    2) Run 
'''

#Libraries
import pandas as pd
import gc
import ml_RunDefinitions as rdefs
import ml_Models as models
import os

#Display options
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.width', 1000)

##### Config ##########

#Run params - cutdown set as in runMachineLearning.py
testName = 'TestA' #Prefix for model
modelName = "NN" # or 'SVR', 'LR', 'Swath'
runList = 'single' #The run definition as per the ml_RunDefitions file

#Load Info  - same as in runMachineLearning.py
storeFolder = '/media/martin/FastData/Data/hdf/timeseries-nn/'   
storeName = 'mlprepared.h5'

### NN params - ignored if LR or SVR selected ###  - same as in runMachineLearning.py
lossFn = "L1"
optimizer = "Adamax"   
iterations = 50
nnLearningRate = 1e-2
scaleY = True

### Additional Load Model parameters ###
modelFolder = "Models-nn-test"
rootPath = "/media/martin/FastData/"
predictionSaveFolder = "/media/martin/FastData/Data/hdf/predictions/"

#Run for all test sets that contain OIB or for manual test set without OIB
withOib = True #If data does not contain OIB then set to false and set the WithoutOib params
#WithoutOib params
noOibKey = 'AllJak11'
noOibData = '/media/martin/FastData/Data/hdf/areas-swathpocajoin/jakobshavn2011.h5'

####### Start of Code ##########

#Model Save info
folderName = testName +'_'+ modelName
saveFolder = rootPath + modelFolder + "/" + folderName

#Prediction save folder
h5Folder = predictionSaveFolder + modelFolder + "/" + folderName


if modelName == "NN":
    addFolder = "_" + lossFn + "_" + optimizer + "_" + str(iterations)
    if scaleY:
        addFolder = addFolder + "_ScaleY/"
    else:
        addFolder = addFolder + "_NoScaleY/"
else:
    addFolder = "/"
    
saveFolder = saveFolder + addFolder
h5Folder = h5Folder + addFolder 

#Columns to Exclude from run
dropCols = ['Lat_Swath','Lon_Swath','X_Swath','Y_Swath','MeanDiffSpread_Swath','DemDiff_Swath','DemDiff_SwathOverPoca','DemDiffMad_Swath','Wf_Number_Swath','StartTime_Swath','Elev_Swath']   

#Get data
store = pd.HDFStore(storeFolder+storeName,mode='r',complevel=9, complib='blosc')
runDefs = rdefs.getRunList(store,runList)

# Loop over all tests
for runKey in runDefs.keys():
    print("########### Running for {} #############".format(runKey))
    runSet = runDefs[runKey]
    testSets = runSet.getTestSets()
        
    saveLocation = saveFolder + runKey + "/"
    
    #Load Model
    model = models.loadModel(modelName,saveLocation)

    #Test result
    if withOib:
        for testKey in testSets.keys():
            testSet = testSets[testKey]
            
            #Prepare test data
            x_test = testSet.drop(['Elev_Oib'],axis=1)
            
            #Predict output
            y_pred = model.predict(x_test) + x_test['Elev_Swath']
            
            #Calculate differences
            resultSet = testSet.copy()
            resultSet.loc[:,'Predicted'] = y_pred
            resultSet.loc[:,'DiffPredicted'] = resultSet['Predicted']-resultSet['Elev_Oib']
            resultSet.loc[:,'DiffSwath'] = resultSet['Elev_Swath']-resultSet['Elev_Oib']
            filteredSet = resultSet[(resultSet['PowerScaled_Swath']>10000) & (resultSet['Coh_Swath'] > 0.8)]
            
            #Save data
            path = h5Folder + runKey + "/"
            if not os.path.exists(path):
                os.makedirs(path)
            resultSet.to_hdf(path + testKey + "_Full.h5","data")
            filteredSet.to_hdf(path + testKey + "_Filtered.h5","data")
    else:
        testKey = noOibKey
        path = h5Folder + runKey + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        #Create stores
        fullStore = pd.HDFStore(path + testKey + "_Full.h5",mode='w',complevel=9, complib='blosc')
        filStore = pd.HDFStore(path + testKey + "_Filtered.h5",mode='w',complevel=9, complib='blosc')
        
        #Load input data
        fname = noOibData
        testStore = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')
        storer = testStore.get_storer('data')
        nrows = storer.nrows
        ii = 0
        increment = 1000000
        
        #Run in increments to ensure fits in memory
        while ii< nrows:
            testSet = storer.read(start=ii,stop=ii+increment)
            x_test = testSet
            
            #Predict output
            y_pred = model.predict(x_test) + x_test['Elev_Swath']
            
            #Calculate differences
            resultSet = testSet.copy()
            resultSet.loc[:,'Predicted'] = y_pred
            filteredSet = resultSet[(resultSet['PowerScaled_Swath']>10000) & (resultSet['Coh_Swath'] > 0.8)]
            
            #Add data to stores
            fullStore.append('data',resultSet,index=False,data_columns=True)
            filStore.append('data',filteredSet,index=False,data_columns=True)
            ii += increment
        
        #Close stores
        fullStore.close()
        filStore.close()
        testStore.close()
    
    #Clear memory    
    del model
    gc.collect()

#Close store
store.close()


