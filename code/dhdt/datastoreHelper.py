#!/usr/bin/env python2

''' Collection of classes and methods that help handle datastore objects '''

__all__ = ['DataStoreHelper']

#Libraries
import gc
import pandas as pd
import numpy as np
import numbers
from datetime import datetime
import glob
import time
import psutil

##### Functions ######

def dateToDayInYear(seconds):
    #Calculate a day in the year from a date
    convert = datetime.fromtimestamp(seconds)
    date = datetime(year=convert.year, month=convert.month, day=convert.day)
    new_year_day = datetime(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1

def dropIf(df,name):
    #Drop column if exist
    if name in df.columns.values:
        return df.drop(columns=[name])
    return df

def fixIf(df,name,filePath):
    #Fix column types if exist
    if name in df.columns.values:
        print('Col fix applied: {}'.format(filePath))
        d = df[~np.isnan(df['wf_number'])]
        d = d[~np.isnan(d['sampleNb'])] 
        d = d.drop(columns=[name])
        d['wf_number'] = d['wf_number'].astype('uint16')
        d['sampleNb'] = d['sampleNb'].astype('uint16')
        d['startTime'] = d['startTime'].astype('int64')
        return d
    return df

def tidyData(df,filePath):
    #Necessary to tidy some of the matlab data - would ideally clean the matlab data in the future    
    d = fixIf(df,'MPTanc',filePath)
    d = dropIf(d,'POCA')
    d = dropIf(d,'readfileopt')
    return d

##### Classes #######

class DataStoreHelper(object):
    ''' Collection of methods that help handle datastore objects '''
    
    @staticmethod
    def createIndex(data,colList=None):
        ''' Creates a fast index of min/max values '''
        
        df = pd.DataFrame({'MinMax': ["Min","Max"]})
        df.set_index("MinMax",inplace=True)
        
        indexList = colList
        #If colList is None then do for all columns
        if colList == None:
            indexList = data.columns.values
        
        for indexCol in indexList:
            if isinstance(data[indexCol][0],numbers.Number):
                minV = min(data[indexCol][data[indexCol].notnull()])
                maxV = max(data[indexCol][data[indexCol].notnull()])
                df[indexCol] = [minV, maxV]
        
        return df
    
    @staticmethod
    def loadData(fname,bbox=None,latLonOrXY=None,dateRange=None):
        '''
        Checks fast index to see if bbox and time window overlaps
        If it does overlap, then it loads and filters the data
        '''
        
        constraints = []
        data = pd.DataFrame()
               
        #Check index 
        indexData = pd.read_hdf(fname,key="index")
        a = indexData
        
        if dateRange != None:
            minDate = time.mktime(datetime.strptime(dateRange[0],"%d/%m/%Y").timetuple())
            maxDate = time.mktime(datetime.strptime(dateRange[1],"%d/%m/%Y").timetuple())+86399 #add number of seconds in day 
            
            #check if any of data is in date range           
            dateCheck = not(a['startTime']['Min']>maxDate or a['startTime']['Max']<minDate)
            if not dateCheck:
                return data
            
             #check if all data in date range
            allInRange = a['startTime']['Min']>=minDate and a['startTime']['Max']<=maxDate
            
            #If all in range then no need to apply constraints - will be much faster loading
            if not allInRange:
                constraints.append('{:s}>={:f}'.format('startTime',minDate))
                constraints.append('{:s}<={:f}'.format('startTime',maxDate))                            
                                       
        
        if bbox != None:
            bMinX = bbox[0]
            bMaxX = bbox[2]
            bMinY = bbox[1]
            bMaxY = bbox[3]
            
            #Set whether using lat/lon or x/y
            x='lon'
            y='lat'
            if latLonOrXY == 'XY':
                x='x'
                y='y'
            
            #check if any of data is in bounding box            
            bboxCheck = not(a[x]['Min']>bMaxX or a[x]['Max']<bMinX or a[y]['Min']>bMaxY or a[y]['Max']<bMinY)
            if not bboxCheck:
                return data            
            
            #check if all data in bounding box
            allInBbox = a[x]['Min']>=bMinX and a[x]['Max']<=bMaxX and a[y]['Min']>=bMinY and a[y]['Max']<=bMaxY
                        
            #If all in bbox then no need to apply constraints - will be much faster loading
            if not allInBbox:
                constraints.append('{:s}>={:f}'.format(x,bMinX))
                constraints.append('{:s}<={:f}'.format(x,bMaxX))
                constraints.append('{:s}>={:f}'.format(y,bMinY))
                constraints.append('{:s}<={:f}'.format(y,bMaxY))
                

        if len(constraints)>0:
            data = pd.read_hdf(fname,key="data",where=constraints)
        else:
            data = pd.read_hdf(fname,key="data")
            
        return data
    
    @staticmethod
    def loadMany(path,bbox=None,latLonOrXY=None,dateRange=None):
        '''
        Checks the fast index of all files in a folder and then loads and appends data
        This appends data in-memory so will hit issues if data is too large
        '''
        
        gc.collect() #Ensure no wasted memory being used before we start
        start = time.time()
        merge = pd.DataFrame()
        i = 0
        matchingCount = 0
        
        #Get file list
        fileList = glob.glob(path)
        numberFiles = len(fileList)
        
        #Load each file, check index and append
        for f in fileList:
            load = DataStoreHelper.loadData(f,bbox,latLonOrXY,dateRange)
            i += 1
            if i%(numberFiles//10) == 0:
                print("Progress: {:.0f}%, Memory: {}%, Swap: {}%".format(i*100.0/numberFiles,psutil.virtual_memory().percent,psutil.swap_memory().percent))
            if load.shape[0] > 0:
                matchingCount+=1
                if merge.shape[0] > 0:
                    merge = pd.concat([merge,load])
                else:
                    merge = load
                    
        merge = merge.reset_index().drop(['index'],axis=1)
        timeTaken = time.time()-start
        print("Done - Matched {} files of {} in {:.2f} seconds".format(matchingCount,numberFiles,timeTaken))
        return merge

    @staticmethod
    def loadManyToHDF(path,store,storeKey,bbox=None,latLonOrXY=None,dateRange=None):
        '''
        Checks the fast index of all files in a folder and then loads and appends data to a persisted HDF5 file
        This appends data on disk so is slower than LoadMany but works for large datasets
        '''
        
        gc.collect() #Ensure no wasted memory being used before we start
        start = time.time()
        i = 0
        matchingCount = 0
        
        #Get file list
        fileList = glob.glob(path)
        numberFiles = len(fileList)
        gcNumber = 0
        
        #Load each file, check index and append
        for f in fileList:
            #Force the garbage collector to run if we've loaded more than 4m rows since last collection
            if gcNumber > 4000000:    
                gc.collect()
                gcNumber = 0
            load = DataStoreHelper.loadData(f,bbox,latLonOrXY,dateRange)
            i += 1
            if i%(numberFiles//10) == 0:
                print("Progress: {:.0f}%, Memory: {}%, Swap: {}%".format(i*100.0/numberFiles,psutil.virtual_memory().percent,psutil.swap_memory().percent))
            length = load.shape[0]
            if length > 1:
                gcNumber += length
                matchingCount += 1
                try:
                    nrows = store.get_storer(storeKey).nrows
                except:
                    nrows = 0
                load = tidyData(load,f)
                load = load.reset_index().drop(['index'],axis=1)
                load.index = pd.Series(load.index) + nrows
                store.append(storeKey,load,index=False,data_columns=True)
        timeTaken = time.time()-start
        print("Done - Matched {} files of {} in {:.2f} seconds".format(matchingCount,numberFiles,timeTaken))    
    
    @staticmethod
    def createFileDateTime(data):
        '''Makes the dataTime string to add to a sharded file'''
        minTime = min(data['startTime'][data['startTime'].notnull()])
        minFormatted = datetime.fromtimestamp(minTime).strftime('%Y%m%d-%H%M%S')
        maxTime = max(data['startTime'][data['startTime'].notnull()])
        maxFormatted = datetime.fromtimestamp(maxTime).strftime('%Y%m%d-%H%M%S')
        return minFormatted + '_' + maxFormatted
       
    @staticmethod
    def loadJoinedArea(fname,uniqueSwath=True,distThreshold=None,maxSwathElev=None,excludeBeforePoca=False,applyNoiseFilter=False,pocaJoinOnly=False):
        '''
        Loads and area store and joins OIB, Swath and POCA
        All columns in Oib, Swath and POCA is in the resultant dataset
        '''
        
        store = pd.HDFStore(fname,mode='r',complevel=9, complib='blosc')
        
        startTime_swath = 'startTime_swath'
        if pocaJoinOnly:
            join = store.get('swath') #This will only work if swath data is not too large
            startTime_swath = 'startTime'
        else:
            #Load Oib and the distance index
            oib = store.get('oib')
            distIndex = store.get('distIndex')
     
            if distThreshold != None:
                distIndex= distIndex[distIndex['distance']<=distThreshold]
           
            #Only use single swath (closest) in the event there are multiple swaths
            if uniqueSwath:
                distIndex = distIndex.loc[distIndex.groupby("swathIndex")["distance"].idxmin()]
                
                
            join = pd.merge(oib,distIndex,left_index=True,right_on='oibIndex',how='inner')
            del oib, distIndex
            
            # Load swath
            gc.collect() #Clear memory
            memSize = psutil.virtual_memory().available/(1024*1024*1024.0)
            chunkSize = int(memSize*0.8*3000000) #Use at most 80% of available memory
            swathStorer = store.get_storer('swath')
            
            iCS = 0
            while iCS < swathStorer.nrows:
                #Join with Swath
                swath = swathStorer.read(start=iCS,stop=iCS+chunkSize)
                joinTemp = pd.merge(join,swath,left_on='swathIndex',right_index=True,how='inner', suffixes=('_oib','_swath'))
                del swath
                if iCS ==0:
                    swathJoin = joinTemp
                else:
                    swathJoin = swathJoin.append(joinTemp)
                del joinTemp
                iCS += chunkSize
                gc.collect() #Keep the memory clean - don't want to risk any issues
            
            join = swathJoin
            del swathJoin
            
            join.sort_values(by='oibIndex',inplace=True)
            join = join.reset_index().drop(['index'],axis=1)
        
        if maxSwathElev != None:
            join = join[join['elev']<=maxSwathElev]
       
                
        #Minimum Cleanup filters
        if applyNoiseFilter:
            join = join[(join['power']>2500) & (join['coh'] > 0.2)]

               
        #Load Poca and join
        poca = store.get('poca')
        join = pd.merge(join,poca,left_on=[startTime_swath,'wf_number'],right_on=['startTime','wf_number'],how='inner', suffixes=('_swath','_poca'))
        del poca
        
        store.close()
        
        if excludeBeforePoca:
            join = join[1.0*join['sampleNb']>=1.0*join['leadEdgeS']] 
        
        gc.collect()
        return join
        
    
    @staticmethod
    def loadJoinedAreaMlReady(fname,uniqueSwath=True,distThreshold=None,maxSwathElev=None,excludeBeforePoca=False,applyNoiseFilter=False):
        '''
        Loads and area store and joins OIB, Swath and POCA
        Outputs only the columns for use with machine learning
        '''
        
        join = DataStoreHelper.loadJoinedArea(fname,uniqueSwath,distThreshold,maxSwathElev,excludeBeforePoca,applyNoiseFilter)
        
        #Select and rename columns of interest
        mlReady = pd.DataFrame({'Elev_Oib' : join['WGS84EllipsoidHeight']
                ,'Lat_Swath' : join['lat_swath']
                ,'Lon_Swath' : join['lon_swath']
                ,'X_Swath' : join['x_swath']
                ,'Y_Swath' : join['y_swath']
                ,'StartTime_Swath' : join['startTime_swath']
                ,'DayInYear_Swath' : join['startTime_swath'].apply(lambda x: dateToDayInYear(x))             
                ,'Heading_Swath' : join['heading_swath']
                ,'Wf_Number_Swath' : join['wf_number']
                ,'LeadEdgeS_Poca' : join['leadEdgeS']
                ,'LeadEdgeW_Poca' : join['leadEdgeW']
                ,'SampleNb_Swath' : join['sampleNb']
                ,'Elev_Swath' : join['elev_swath']
                ,'Coh_Swath' : join['coh_swath']
                ,'Phase_Swath' : join['phase_swath']
                ,'PhaseConfidence_Swath' : join['phaseConfidence']
                ,'PhaseSSegment_Swath' : join['phaseSSegment']
                ,'PowerScaled_Swath' : join['power_swath']
                ,'PowerWatt_Swath' : join['power_2']
                ,'DemDiff_Swath' : join['demDiff_swath']
                ,'DemDiffMad_Swath' : join['demDiffMad_swath']
                ,'MeanDiffSpread_Swath' : join['meanDiffSpread_swath']
                ,'SampleNb_SwathMinusLeadEdgeS' : 1.0*join['sampleNb']-1.0*join['leadEdgeS']
                ,'Coh_SwathOverPoca' : join['coh_swath']/join['coh_poca']
                ,'Phase_SwathOverPoca' : join['phase_swath']/join['phase_poca']
                ,'PowerScaled_SwathOverPoca' : join['power_swath']/join['power_poca']
                ,'DemDiff_SwathOverPoca' : join['demDiff_swath']/join['demDiff_poca']
                ,'Dist_SwathToPoca' : np.sqrt((join['x_swath']-join['x'])**2 + (join['y_swath']-join['y'])**2)
                })
        del join
        gc.collect()
        return mlReady
    
    @staticmethod
    def loadJoinedAreaMlPredict(fname,maxSwathElev=None,excludeBeforePoca=False,applyNoiseFilter=False):
        '''
        Loads and area store and joins Swath and POCA only - OIB excluded
        Outputs only the columns for use to apply an existing ML model to predict elevation adjustment
        '''
        
        join = DataStoreHelper.loadJoinedArea(fname,maxSwathElev=maxSwathElev,excludeBeforePoca=excludeBeforePoca,applyNoiseFilter=applyNoiseFilter,pocaJoinOnly=True)
        
        #Select and rename Swath and POCA columns of interest
        mlReady = pd.DataFrame({'Lat_Swath' : join['lat_swath']
                ,'Lon_Swath' : join['lon_swath']
                ,'X_Swath' : join['x_swath']
                ,'Y_Swath' : join['y_swath']
                ,'StartTime_Swath' : join['startTime']
                ,'DayInYear_Swath' : join['startTime'].apply(lambda x: dateToDayInYear(x))             
                ,'Heading_Swath' : join['heading_swath']
                ,'Wf_Number_Swath' : join['wf_number']
                ,'LeadEdgeS_Poca' : join['leadEdgeS']
                ,'LeadEdgeW_Poca' : join['leadEdgeW']
                ,'SampleNb_Swath' : join['sampleNb']
                ,'Elev_Swath' : join['elev_swath']
                ,'Coh_Swath' : join['coh_swath']
                ,'Phase_Swath' : join['phase_swath']
                ,'PhaseConfidence_Swath' : join['phaseConfidence']
                ,'PhaseSSegment_Swath' : join['phaseSSegment']
                ,'PowerScaled_Swath' : join['power_swath']
                ,'PowerWatt_Swath' : join['power_2']
                ,'DemDiff_Swath' : join['demDiff_swath']
                ,'DemDiffMad_Swath' : join['demDiffMad_swath']
                ,'MeanDiffSpread_Swath' : join['meanDiffSpread_swath']
                ,'SampleNb_SwathMinusLeadEdgeS' : 1.0*join['sampleNb']-1.0*join['leadEdgeS']
                ,'Coh_SwathOverPoca' : join['coh_swath']/join['coh_poca']
                ,'Phase_SwathOverPoca' : join['phase_swath']/join['phase_poca']
                ,'PowerScaled_SwathOverPoca' : join['power_swath']/join['power_poca']
                ,'DemDiff_SwathOverPoca' : join['demDiff_swath']/join['demDiff_poca']
                ,'Dist_SwathToPoca' : np.sqrt((join['x_swath']-join['x_poca'])**2 + (join['y_swath']-join['y_poca'])**2)
                })
        del join
        gc.collect()
        return mlReady
        