import scipy as sp
from scipy.interpolate import griddata 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc


############## Functions #######################

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)


def plotAndSave(data,dataName,metric,savePath,year,xTrainA,yTrainA,xTrainB,yTrainB,cmap,mask=[],vmin=None,vmax=None,intervals=51,saveDataFile=False,elevLabel=""):
    
    piv = data.pivot(index='Y_Round',columns='X_Round',values=metric)
    
    xi = piv.columns.values
    yi = piv.index.values
    zi = piv.values
    
    if mask!=[]:
        zi = np.ma.masked_where(1-np.flip(mask,axis=0),zi)
    
    
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
    #cbar.ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
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
    #np.savetxt(savePath+title+'-NoRowColHeaders.csv',zi,delimiter=',')


def groupData(data):   
    
    dataSmall = data[['X_Swath','Y_Swath','Predicted','Elev_Swath']]
    dataSmall.loc[:,'Predicted-Swath']=dataSmall['Predicted']-dataSmall['Elev_Swath']
    dataSmall.loc[:,'X_Round']=(dataSmall['X_Swath']/gridWidth).round()*gridWidth
    dataSmall.loc[:,'Y_Round']=(dataSmall['Y_Swath']/gridWidth).round()*gridWidth
    dataSmall.drop(['X_Swath','Y_Swath'],axis=1,inplace=True)
    
    
    grouped = dataSmall.groupby(['X_Round','Y_Round'],as_index=False)
    
    return grouped.mean()
    
    '''
    #Mean
    meanD = grouped.mean()
    plotAndSave(meanD,dataName,'Predicted','DEM(Mean)',savePath,year,plt.cm.jet,mask=mask)
    plotAndSave(meanD,dataName,'Elev_Swath','DEM(Mean)',savePath,year,plt.cm.jet,mask=mask)
    plotAndSave(meanD,dataName,'Predicted-Swath','Diff(Mean)',savePath,year,plt.cm.bwr,centre=True,mask=mask)

    #Median
    medianD = grouped.median()
    plotAndSave(medianD,dataName,'Predicted','DEM(Median)',savePath,year,plt.cm.jet,mask=mask)
    plotAndSave(medianD,dataName,'Elev_Swath','DEM(Median)',savePath,year,plt.cm.jet,mask=mask)
    plotAndSave(medianD,dataName,'Predicted-Swath','Diff(Median)',savePath,year,plt.cm.bwr,centre=True,mask=mask)
    
    #Density
    countD = grouped.count()
    plotAndSave(countD,dataName,'Predicted','Density',savePath,year,plt.cm.Greens,mask=mask)
    #plotAndSave(countD,dataName,'Elev_Swath','Density',savePath,year,plt.cm.jet)
    '''


######### Config ###############
#fname = '/media/martin/FastData/Data/hdf/predictions/Models-nn-5k/BigRunNN_NN_Huber_Adamax_5000_NoScaleY/jak11train/jak11test_Full.h5'


gridWidth = 500.0
saveDataFile = False
fullOrFiltered = 'Full'

#Everything
#label = 'FullArea'
#xmin = -260000
#ymin = -2370000
#xmax = -110000
#ymax = -2150000

#Larger Area
#label = 'MediumArea'
#xmin = -200000
#ymin = -2325000
#xmax = -137500
#ymax = -2250000

#Small Area
label = 'SmallArea'
xmin = -170000#np.min(x)
ymin = -2280000#np.min(y)
xmax = -137500#np.max(x)
ymax = -2260000#np.max(y)

#Mask###
mask = []
#mask = np.loadtxt('/media/martin/FastData/Plots/Masks/GimpIceMask_500m_JakobModified_MLzoomSmall.csv',delimiter=',')
#mask = np.loadtxt('/media/martin/FastData/Plots/Masks/GimpIceMask_500m_JakobModified_MLzoomLarge.csv',delimiter=',')


##### Repeat for Full ######
area = 'all11to14'
test = 'jak11'
fnameFull = '/media/martin/FastData/Data/hdf/predictions/Models-nn-mod/HSmallRun_NN_L1_Adamax_50000_ScaleY/{}/All{}_{}.h5'.format(area,test.title(),fullOrFiltered)
dataFull= pd.read_hdf(fnameFull,key="data")
dataFull = dataFull[(dataFull['X_Swath']>xmin) & (dataFull['X_Swath']<xmax) & (dataFull['Y_Swath']>ymin) & (dataFull['Y_Swath']<ymax) ]
dataFull = dataFull[dataFull['Predicted']>0]
dataFull = dataFull[dataFull['Predicted']<(np.max(dataFull['Elev_Swath'])+100)]
dataA = groupData(dataFull)

fnameTrain = '/media/martin/FastData/Data/hdf/predictions/Models-nn-compare/CompRun_NN_Huber_Adamax_50000_NoScaleY/{}/{}_Full.h5'.format(test,test)
dataTrain = pd.read_hdf(fnameTrain,key="data")
dataTrain = dataTrain[(dataTrain['X_Swath']>xmin) & (dataTrain['X_Swath']<xmax) & (dataTrain['Y_Swath']>ymin) & (dataTrain['Y_Swath']<ymax) ]
xTrainA = dataTrain['X_Swath'].values
yTrainA = dataTrain['Y_Swath'].values

del dataFull, dataTrain
gc.collect()

area = 'all11to14'
test = 'jak12'
fnameFull = '/media/martin/FastData/Data/hdf/predictions/Models-nn-mod/HSmallRun_NN_L1_Adamax_50000_ScaleY/{}/All{}_{}.h5'.format(area,test.title(),fullOrFiltered)
dataFull= pd.read_hdf(fnameFull,key="data")
dataFull = dataFull[(dataFull['X_Swath']>xmin) & (dataFull['X_Swath']<xmax) & (dataFull['Y_Swath']>ymin) & (dataFull['Y_Swath']<ymax) ]
dataFull = dataFull[dataFull['Predicted']>0]
dataFull = dataFull[dataFull['Predicted']<(np.max(dataFull['Elev_Swath'])+100)]
dataB = groupData(dataFull)

fnameTrain = '/media/martin/FastData/Data/hdf/predictions/Models-nn-compare/CompRun_NN_Huber_Adamax_50000_NoScaleY/{}/{}_Full.h5'.format(test,test)
dataTrain = pd.read_hdf(fnameTrain,key="data")
dataTrain = dataTrain[(dataTrain['X_Swath']>xmin) & (dataTrain['X_Swath']<xmax) & (dataTrain['Y_Swath']>ymin) & (dataTrain['Y_Swath']<ymax) ]
xTrainB = dataTrain['X_Swath'].values
yTrainB = dataTrain['Y_Swath'].values

del dataFull, dataTrain
gc.collect()

dataComb = pd.merge(dataA,dataB,left_on=['X_Round','Y_Round'],right_on=['X_Round','Y_Round'],how='inner', suffixes=('_A','_B'))

dataComb.loc[:,'dhdt-Predicted']=dataComb['Predicted_B']-dataComb['Predicted_A']
dataComb.loc[:,'dhdt-Elev_Swath']=dataComb['Elev_Swath_B']-dataComb['Elev_Swath_A']
dataComb.loc[:,'dhdt-Predicted-Elev_Swath']=dataComb['dhdt-Predicted']-dataComb['dhdt-Elev_Swath']

labelPlot = label + '-2011'
plotAndSave(dataA,fullOrFiltered,'Predicted','/media/martin/FastData/Plots/',labelPlot,xTrainA,yTrainA,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataA,fullOrFiltered,'Elev_Swath','/media/martin/FastData/Plots/',labelPlot,xTrainA,yTrainA,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataA,fullOrFiltered,'Predicted-Swath','/media/martin/FastData/Plots/',labelPlot,xTrainA,yTrainA,[],[],plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Difference')

labelPlot = label + '-2012'
plotAndSave(dataB,fullOrFiltered,'Predicted','/media/martin/FastData/Plots/',labelPlot,xTrainB,yTrainB,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataB,fullOrFiltered,'Elev_Swath','/media/martin/FastData/Plots/',labelPlot,xTrainB,yTrainB,[],[],plt.cm.jet,mask=mask,vmin=0,vmax=1500,saveDataFile=saveDataFile)
plotAndSave(dataB,fullOrFiltered,'Predicted-Swath','/media/martin/FastData/Plots/',labelPlot,xTrainB,yTrainB,[],[],plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Difference')

labelPlot = label + '-2011to12'
plotAndSave(dataComb,fullOrFiltered,'dhdt-Predicted','/media/martin/FastData/Plots/',labelPlot,xTrainA,yTrainA,xTrainB,yTrainB,plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Change')
plotAndSave(dataComb,fullOrFiltered,'dhdt-Elev_Swath','/media/martin/FastData/Plots/',labelPlot,xTrainA,yTrainA,xTrainB,yTrainB,plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Change')
plotAndSave(dataComb,fullOrFiltered,'dhdt-Predicted-Elev_Swath','/media/martin/FastData/Plots/',labelPlot,xTrainA,yTrainA,xTrainB,yTrainB,plt.cm.bwr,mask=mask,vmin=-50,vmax=50,saveDataFile=saveDataFile,elevLabel='Change Difference')

#plotGrids(dataA,dataB, 'Full','/media/martin/FastData/Plots/',xTrain,yTrain,year,mask)

#del dataFull
#gc.collect()



