#!/usr/bin/env python2

'''
Performs the machine learning
Saves the models and result analysis

Instructions:
    1) Set configuration section below
    2) Run 
'''

#Libraries
import pandas as pd
import time
from datetime import datetime
import gc
import ml_learningResults as res
import ml_RunDefinitions as rdefs
import ml_Models as models

#Display options
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.width', 1000)


###### Config ########

#Run params 
testName = 'TestA' #Prefix to add to model
modelName = "NN" # or 'SVR', 'LR', 'Swath'
test = False #if set, will append timestamp to outer folder
loadOnly = False #If set, will load prerun models and regenerate analysis
runList = 'small' #The run definition as per the ml_RunDefitions file

#Save Model Folder
saveModelFolder = "/media/martin/FastData/Models-nn-test/"

#Load Info
storeFolder = '/media/martin/FastData/Data/hdf/timeseries-nn/'   
storeName = 'mlprepared.h5'

### NN params - ignored if LR or SVR selected ###
lossFn = "L1"
optimizer = "Adamax"   
iterations = 5000
nnLearningRate = 1e-2
scaleY = True

####### Start of Code ##########

#Save info
folderName = testName +'_'+ modelName
if test:
    folderName = folderName +'_' +datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
saveFolder = saveModelFolder + folderName

if modelName == "NN":
    saveFolder = saveFolder + "_" + lossFn + "_" + optimizer + "_" + str(iterations)
    if scaleY:
        saveFolder = saveFolder + "_ScaleY/"
    else:
        saveFolder = saveFolder + "_NoScaleY/"
else:
    saveFolder = saveFolder +"/"

#Columns to Exclude from run
dropCols = ['Lat_Swath','Lon_Swath','X_Swath','Y_Swath','MeanDiffSpread_Swath','DemDiff_Swath','DemDiff_SwathOverPoca','DemDiffMad_Swath','Wf_Number_Swath','StartTime_Swath','Elev_Swath']   

#Get data
store = pd.HDFStore(storeFolder+storeName,mode='r',complevel=9, complib='blosc')
runDefs = rdefs.getRunList(store,runList)

#Initialise Result collection
resultsFull = res.ResultCollection()
resultsFil = res.ResultCollection()

# Loop over all tests
for runKey in runDefs.keys():
    print("########### Running for {} #############".format(runKey))
    runSet = runDefs[runKey]
    trainName, trainSet = runSet.getTrainSet()
    testSets = runSet.getTestSets()
    
    #Prepare train/test data
    x_train = trainSet.drop(['Elev_Oib'],axis=1)
    y_train = trainSet.Elev_Oib
    y_trainElevDiff = y_train - x_train['Elev_Swath']
    
    saveLocation = saveFolder + runKey + "/"
    #Train Model
    if loadOnly:
        model = models.loadModel(modelName,saveLocation)
    else:
        model = models.runModel(modelName,x_train,y_trainElevDiff,dropCols,saveLocation,lr=nnLearningRate,maxIter=iterations,lossFn=lossFn,optimizer=optimizer,scaleY=scaleY)
    
    #Test result
    for testKey in testSets.keys():
        testSet = testSets[testKey]
        x_test = testSet.drop(['Elev_Oib'],axis=1)
        
        #Predict output
        y_pred = model.predict(x_test) + x_test['Elev_Swath']
        
        #Compare output to expected
        resultSet = testSet.copy()
        resultSet.loc[:,'Predicted'] = y_pred
        resultSet.loc[:,'Difference'] = resultSet['Predicted']-resultSet['Elev_Oib']
        
        #Create analysis files
        filteredSet = resultSet[(resultSet['PowerScaled_Swath']>10000) & (resultSet['Coh_Swath'] > 0.8)]
        resultsFull.add(runKey,testKey,resultSet,model.TimeTaken)
        resultsFil.add(runKey,testKey,filteredSet,model.TimeTaken)
    
    #Clear memory    
    del model
    gc.collect()

#Create resultName
resultName = modelName
if modelName == "NN":
    resultName = resultName +"_" + optimizer + "_" + lossFn
    if scaleY:
        resultName = resultName + "_ScaleY"
    else:
        resultName = resultName + "_NoScaleY"

#Save results
print("Full Results")
print(resultsFull)
resultsFull.save(saveFolder+"Results",resultName+"_Full")#+fileName)

print("\nFiltered Results")
print(resultsFil)
resultsFil.save(saveFolder+"Results",resultName+"_Filtered")#+fileName)

store.close()