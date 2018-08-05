import matplotlib.pyplot as plt
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
import ml_ModelsMod as models


pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.width', 1000)

#### Functions ######



    


##### Start of Code ##########


#Run params 
testName = 'BatchNorm'
modelName = "NN"
test = False
loadOnly = False
runList = 'small'

#Load Info
storeFolder = '/media/martin/FastData/Data/hdf/timeseries-nn/'   
storeName = 'mlprepared.h5'

####NN params - ignored if LR or SVR selected###
lossFn = "L1"
optimizer = "Adamax"   
iterations = 50000
nnLearningRate = 1e-2
scaleY = True 

#Save info
folderName = testName +'_'+ modelName
if test:
    folderName = folderName +'_' +datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
saveFolder = "/media/martin/FastData/Models-nn-mod/"+folderName

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
    
    trainSet = trainSet.sample(frac=1).reset_index(drop=True)
    
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
        y_pred = model.predict(x_test) + x_test['Elev_Swath']
        resultSet = testSet.copy()
        resultSet.loc[:,'Predicted'] = y_pred
        resultSet.loc[:,'Difference'] = resultSet['Predicted']-resultSet['Elev_Oib']
        filteredSet = resultSet[(resultSet['PowerScaled_Swath']>10000) & (resultSet['Coh_Swath'] > 0.8)]
        resultsFull.add(runKey,testKey,resultSet,model.TimeTaken)
        resultsFil.add(runKey,testKey,filteredSet,model.TimeTaken)
    
    #Clear memory    
    del model
    gc.collect()


resultName = modelName
if modelName == "NN":
    resultName = resultName +"_" + optimizer + "_" + lossFn
    if scaleY:
        resultName = resultName + "_ScaleY"
    else:
        resultName = resultName + "_NoScaleY"

print("Full Results")
print(resultsFull)
resultsFull.save(saveFolder+"Results",resultName+"_Full")#+fileName)

print("\nFiltered Results")
print(resultsFil)
resultsFil.save(saveFolder+"Results",resultName+"_Filtered")#+fileName)

store.close()



'''
y_testElevDiff = y_test - x_test['Elev_Swath']



#Columns to drop when running on full data - have left included DayInYear_Swath and StartTime_Swath
dropCols = ['Lat_Swath','Lon_Swath','X_Swath','Y_Swath','MeanDiffSpread_Swath','DemDiff_Swath','DemDiff_SwathOverPoca','DemDiffMad_Swath','Wf_Number_Swath','StartTime_Swath','Elev_Swath']   



print("Data Size Proportion = {} ".format(dataSize))    

##### No Learning ########




results.add('Arctic DEM',x_test.Elev_Swath-x_test.DemDiff_Swath,y_test,x_test,0)


#### Test #######
reg = ll.RegressionScikit()
reg.load(saveFolder+"Scikit/"+storeName)
y_predicted = reg.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('TestReg Diff Elev Scale',y_predSet,y_testSet,x_test,reg.TimeTaken)



###### NN Diff ###############


print("Starting NN - Diff at {}".format(time.ctime()))
lossFn = "Huber"
optimizer = "Adamax"    
nnElevDiff = ll.RegressionTorch()
nnElevDiff.train(x_train,y_trainElevDiff,dropCols,iterations,lossFn=lossFn,learningRate = nnLearningRate,optimizer=optimizer,scaleY=False)
y_predicted = nnElevDiff.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('NN Elev Diff',y_predSet,y_testSet,x_test,nnElevDiff.TimeTaken)
storeName = fileName +'-Adamax-Diff'+'-maxIts='+str(iterations)+'-Leaky-Elev' 
nnElevDiff.save(saveFolder+"Pytorch/"+storeName)


#### Test #######
reg = ll.RegressionTorch()
reg.load(saveFolder+"st11train/")
testSet= store['st11test']
x_test = testSet.drop(['Elev_Oib'],axis=1)
y_predicted = reg.predict(x_test)
y_pred = y_predicted + testSet['Elev_Swath']
resultSet = testSet.copy()
resultSet.loc[:,'Predicted'] = y_pred
resultSet.loc[:,'Difference'] = resultSet['Predicted']-resultSet['Elev_Oib']
filteredSet = resultSet[(resultSet['PowerScaled_Swath']>10000) & (resultSet['Coh_Swath'] > 0.8)]
resultsFull.add("a","a",resultSet,0)
resultsFil.add("b","b",filteredSet,0)

###### NN Diff Scaled ###############

print("Starting NN - Diff Scaled at {}".format(time.ctime()))
lossFn = "Huber"
optimizer = "Adamax"    
nnElevDiffScale = ll.RegressionTorch()
nnElevDiffScale.train(x_train,y_trainElevDiff,dropCols,iterations,lossFn=lossFn,learningRate = nnLearningRate,optimizer=optimizer)
y_predicted = nnElevDiffScale.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('NN Elev Diff Scaled',y_predSet,y_testSet,x_test,nnElevDiffScale.TimeTaken)
storeName = fileName +'-Adamax-Diff-Scale'+'-maxIts='+str(iterations)+'-Leaky-Elev' 
nnElevDiffScale.save(saveFolder+"Pytorch/"+storeName)

#### Test #######
reg = ll.RegressionTorch()
reg.load(saveFolder+"Pytorch/"+storeName)
y_predicted = reg.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('Test Diff Elev Scale',y_predSet,y_testSet,x_test,reg.TimeTaken)


###### SVR - Diff ##########

print("Starting SVR - Diff at {}".format(time.ctime())) 
model = SVR(kernel='rbf',epsilon=1e-3,C=1e2)
reg = ll.RegressionScikit()
reg.train(x_train,y_trainElevDiff,model,dropCols)
y_predicted = reg.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('SVR Diff Elev Scale',y_predSet,y_testSet,x_test,reg.TimeTaken)
storeName = fileName +'-SVR-Diff-Scale' 
reg.save(saveFolder+"Scikit/"+storeName)


#### Test #######
reg2 = ll.RegressionScikit()
reg2.load(saveFolder+"Scikit/"+storeName)
y_predicted = reg2.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('T3 Diff Elev Scale',y_predSet,y_testSet,x_test,reg2.TimeTaken)


###### NN Diff Scaled ###############

print("MSE - Diff Scaled at {}".format(time.ctime()))
lossFn = "MSE"
optimizer = "Adamax"    
nnElevDiffScale = ll.RegressionTorch()
nnElevDiffScale.train(x_train,y_trainElevDiff,dropCols,iterations,lossFn=lossFn,learningRate = nnLearningRate,optimizer=optimizer)
y_predicted = nnElevDiffScale.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('MSE Elev Diff Scaled',y_predSet,y_testSet,x_test,nnElevDiffScale.TimeTaken)
storeName = fileName +'-MSE-Diff-Scale'+'-maxIts='+str(iterations)+'-Leaky-Elev' 
nnElevDiffScale.save(saveFolder+"Pytorch/"+storeName)

###### NN Diff Scaled ###############

print("Adam - Diff Scaled at {}".format(time.ctime()))
lossFn = "Huber"
optimizer = "Adam"    
nnElevDiffScale = ll.RegressionTorch()
nnElevDiffScale.train(x_train,y_trainElevDiff,dropCols,iterations,lossFn=lossFn,learningRate = nnLearningRate,optimizer=optimizer)
y_predicted = nnElevDiffScale.predict(x_test)
y_predSet = y_predicted + x_test['Elev_Swath']
y_testSet = y_test
results.add('Adam Elev Diff Scaled',y_predSet,y_testSet,x_test,nnElevDiffScale.TimeTaken)
storeName = fileName +'-Adam-Diff-Scale'+'-maxIts='+str(iterations)+'-Leaky-Elev' 
nnElevDiffScale.save(saveFolder+"Pytorch/"+storeName)

print("Done at {}".format(time.ctime())) 

########### Summary Results ###############
results.addSummary(y_train,y_test)

########### Print Summary ###################

print("\nSummary of Data")
print("------------------------")
print("OIB data")
print(results.summary())
print("------------------------")
'''

