#!/usr/bin/env python2

''' Creates a DEM and DHDT plot

Instructions:
    1) Set configuration section below
    2) Run
    
'''

#Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc


######### Config ###############

gridWidth = 500.0 #resolution of the grid
saveDataFile = False #Output data files as well as plots - will be large
fullOrFiltered = 'Full' # or 'Filtered'
trainSet = 'all11to14'
testSetA = 'jak11'
testSetB = 'jak12'
yearA = 2011
yearB = 2012
dhdtLabel = '2011to12'

#Folders
modelPathWholeArea = '/media/martin/FastData/Data/hdf/predictions/Models-nn-mod/HSmallRun_NN_L1_Adamax_50000_ScaleY'
modelPathTrainingPoints = '/media/martin/FastData/Data/hdf/predictions/Models-nn-compare/CompRun_NN_Huber_Adamax_50000_NoScaleY'
plotSaveFolder = '/media/martin/FastData/Plots/'


#Everything - saved for reference
#label = 'FullArea'
#xmin = -260000
#ymin = -2370000
#xmax = -110000
#ymax = -2150000
#mask = np.loadtxt('/media/martin/FastData/Plots/Masks/GimpIceMask_500m_JakobModified_MLzoomLarge.csv',delimiter=',')

#Larger Area - saved for reference
#label = 'MediumArea'
#xmin = -200000
#ymin = -2325000
#xmax = -137500
#ymax = -2250000
#mask = np.loadtxt('/media/martin/FastData/Plots/Masks/GimpIceMask_500m_JakobModified_MLzoomSmall.csv',delimiter=',')

#Small Area
label = 'SmallArea'
xmin = -170000
ymin = -2280000
xmax = -137500
ymax = -2260000
mask = []



############## Display Settings #######################

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

############## Functions #######################

def plotAndSave(data,dataName,metric,savePath,year,xTrainA,yTrainA,xTrainB,yTrainB,cmap,mask=[],vmin=None,vmax=None,intervals=51,saveDataFile=False,elevLabel=""):
    ''' creates plot and saves output png '''
    
    #Get x,y,z values
    piv = data.pivot(index='Y_Round',columns='X_Round',values=metric)
    xi = piv.columns.values
    yi = piv.index.values
    zi = piv.values
    
    #Apply mask
    if mask!=[]:
        zi = np.ma.masked_where(1-np.flip(mask,axis=0),zi)
    
    #create contour map
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.contourf(xi,yi,zi,intervals,cmap=cmap,vmin=vmin,vmax=vmax)
    
    #Plot Training paths
    ax.scatter(xTrainA,yTrainA,color='k',marker='.',s=7)
    if len(xTrainB) != 0:
        ax.scatter(xTrainB,yTrainB,color='g',marker='.',s=7)
    
    #Define colourbar
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(zi)
    m.set_clim(vmin, vmax)
    msize = (vmax-vmin)/intervals
    cbar = fig.colorbar(m,boundaries=np.arange(vmin,vmax+.1,msize))
    cbar.ax.set_ylabel('Elevation {} (m)'.format(elevLabel))
    
    #Format x axis
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.set_xlim(-170000, -138000) 
    plt.ylabel('y', figure=fig)
    plt.xlabel('x', figure=fig)
    
    #Add title
    title = '{}-{}({})'.format(year,metric,dataName)
    plt.title(title)

    #Save figure
    fig.savefig(savePath+title+'.png',format='png')
    if saveDataFile:
        dataOut = pd.DataFrame(data=zi,index=yi,columns=xi)
        dataOut.to_csv(savePath+title+'-WithRowColHeaders.csv')


def groupData(data,gridWidth):   
    ''' Groups data into specified resolution '''
    
    #Select columns needed for grid
    dataSmall = data[['X_Swath','Y_Swath','Predicted','Elev_Swath']]
    dataSmall.loc[:,'Predicted-Swath']=dataSmall['Predicted']-dataSmall['Elev_Swath']
    
    #Round and group data to match grid width
    dataSmall.loc[:,'X_Round']=(dataSmall['X_Swath']/gridWidth).round()*gridWidth
    dataSmall.loc[:,'Y_Round']=(dataSmall['Y_Swath']/gridWidth).round()*gridWidth
    dataSmall.drop(['X_Swath','Y_Swath'],axis=1,inplace=True)
    grouped = dataSmall.groupby(['X_Round','Y_Round'],as_index=False)
    
    return grouped.mean()

def createGrid(modelPathWholeArea,area,test,gridWidth,fullOrFiltered,xmin,xmax,ymin,ymax):
    ''' Create gridded data set '''
    
    #Load whole area data
    fnameFull = '{}/{}/All{}_{}.h5'.format(modelPathWholeArea,area,test.title(),fullOrFiltered)
    dataFull= pd.read_hdf(fnameFull,key="data")
    
    #Filter for bounding box
    dataFull = dataFull[(dataFull['X_Swath']>xmin) & (dataFull['X_Swath']<xmax) & (dataFull['Y_Swath']>ymin) & (dataFull['Y_Swath']<ymax) ]
    
    #Clean data
    dataFull = dataFull[dataFull['Predicted']>0]
    dataFull = dataFull[dataFull['Predicted']<(np.max(dataFull['Elev_Swath'])+100)]
    
    #Group data
    dataGrouped = groupData(dataFull,gridWidth)
    
    #Clear memory
    del dataFull
    gc.collect()
    
    return dataGrouped

def getTrainPoints(modelPathTrainingPoints,test,xmin,xmax,ymin,ymax):
    ''' Get training data set '''
    
     #Load training data
    fnameTrain = '{}/{}/{}_Full.h5'.format(modelPathTrainingPoints,test,test)
    dataTrain = pd.read_hdf(fnameTrain,key="data")
    
    #Filter for bounding box
    dataTrain = dataTrain[(dataTrain['X_Swath']>xmin) & (dataTrain['X_Swath']<xmax) & (dataTrain['Y_Swath']>ymin) & (dataTrain['Y_Swath']<ymax) ]
    
    #Get x, y points of training set
    xTrain = dataTrain['X_Swath'].values
    yTrain = dataTrain['Y_Swath'].values
    
    #Clear memory
    del dataTrain
    gc.collect()
    return xTrain, yTrain


############## Main Code #######################

# Load year A data
dataA = createGrid(modelPathWholeArea,trainSet,testSetA,gridWidth,fullOrFiltered,xmin,xmax,ymin,ymax)
xTrainA, yTrainA = getTrainPoints(modelPathTrainingPoints,testSetA,xmin,xmax,ymin,ymax)

# Load year B data
dataB = createGrid(modelPathWholeArea,trainSet,testSetB,gridWidth,fullOrFiltered,xmin,xmax,ymin,ymax)
xTrainB, yTrainB = getTrainPoints(modelPathTrainingPoints,testSetB,xmin,xmax,ymin,ymax)

#Calculate dhdt
dataComb = pd.merge(dataA,dataB,left_on=['X_Round','Y_Round'],right_on=['X_Round','Y_Round'],how='inner', suffixes=('_A','_B'))
dataComb.loc[:,'dhdt-Predicted']=dataComb['Predicted_B']-dataComb['Predicted_A']
dataComb.loc[:,'dhdt-Elev_Swath']=dataComb['Elev_Swath_B']-dataComb['Elev_Swath_A']
dataComb.loc[:,'dhdt-Predicted-Elev_Swath']=dataComb['dhdt-Predicted']-dataComb['dhdt-Elev_Swath']

#Create plots - Year A
labelPlot = label + '-' + str(yearA)
plotAndSave(dataA,fullOrFiltered,'Predicted',plotSaveFolder,labelPlot,xTrainA,yTrainA,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataA,fullOrFiltered,'Elev_Swath',plotSaveFolder,labelPlot,xTrainA,yTrainA,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataA,fullOrFiltered,'Predicted-Swath',plotSaveFolder,labelPlot,xTrainA,yTrainA,[],[],plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Difference')

#Create plots - Year B
labelPlot = label + '-' + str(yearB)
plotAndSave(dataB,fullOrFiltered,'Predicted',plotSaveFolder,labelPlot,xTrainB,yTrainB,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataB,fullOrFiltered,'Elev_Swath',plotSaveFolder,labelPlot,xTrainB,yTrainB,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataB,fullOrFiltered,'Predicted-Swath',plotSaveFolder,labelPlot,xTrainB,yTrainB,[],[],plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Difference')

#Create plots - Dhdt
labelPlot = label + dhdtLabel
plotAndSave(dataComb,fullOrFiltered,'dhdt-Predicted',plotSaveFolder,labelPlot,xTrainA,yTrainA,xTrainB,yTrainB,plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Change')
plotAndSave(dataComb,fullOrFiltered,'dhdt-Elev_Swath',plotSaveFolder,labelPlot,xTrainA,yTrainA,xTrainB,yTrainB,plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Change')
plotAndSave(dataComb,fullOrFiltered,'dhdt-Predicted-Elev_Swath',plotSaveFolder,labelPlot,xTrainA,yTrainA,xTrainB,yTrainB,plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Change Difference')




