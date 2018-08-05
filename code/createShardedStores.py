#!/usr/bin/env python2

''' Convert Matlab and CSV files to sharded HDF5 data files

Instructions:
    1) Set configuration section below
    2) Run with a dhdt config - e.g. cryomlt/configs/projection.cfg
    
'''

#Libraries
import dhdt
import argparse
import time
import glob
from dhdt.geoWrapper import GeoWrapper
from dhdt.datastoreHelper import DataStoreHelper
import pandas as pd

def main():
    ''' main entry point to code '''
    
    ########## Config ##########
    
    #Set test flag if wish to generate single matlab or csv file to test config ok.
    test = False
    
    converter = 'Matlab' #Or 'CSV'

    #Optional bbox in wgs84 coordinates [bottom left to top right]
    #bbox = [81.4,-96.734,81.41,-96.73]
    bbox = None
    
    #Matlab or CSV file pattern match
    fileglob = '/media/martin/DATA/Data/MatFiles/Swath/2012/*1B_201202*.mat'
    
    #Location to store shards
    storeFolder = '/media/martin/DATA/Data/hdf/swath/2012/'
    
    #Shard prefix, so can easily identify. Put 'swath_', 'poca_' or 'oib_' or anything else
    filePrefix = 'swath_'
    
    #Set data lengths - want large for swath (500k, small for poca and oib)
    recordLength = 500000
    recordBuffer = 150000 #Maximum over recordLength to allow in file
    recordMin = 400000 #Minimum to allow in file - will concatenate files if below this
    
    #Alternate for Poca or OIB
    #recordLength = 6000
    #recordBuffer = 2000
    #recordMin = 4000
    
    ######### Additional DHDT Config ###########
    
    #This is dhdt config parsing code. I have not modified
    
    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--rebuild-store',action='store_true',default=False,help="rebuild data store even though store is newer than the input files")
    args = parser.parse_args()
    dhdt.initLog(args)

    if args.submit == 'sge':
        batch = dhdt.SGEProcess(args)
    elif args.submit == 'pbs':
        batch = dhdt.PBSProcess(args)
    else:
        batch = None

    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(args.config)
  

    ############# Start of code ##############
    
    start = time.time()
    
    #Define reader
    if converter == 'Matlab':
        reader = dhdt.dataReader.MatlabReader()
    else:
        reader = dhdt.dataReader.CSVReader()

    #Temporary variables
    hold = False
    holdData = None
    
    #Get file list in order
    fileList = glob.glob(fileglob)
    fileList.sort()
    
    #Set counters
    ii = 0
    fileCount = len(fileList)
    
    #Iterate through each file and convert to shards
    for d in fileList:
        
        ii += 1
        print('Processing file {} of {}: {}'.format(ii,fileCount,d))
        
        loadData = reader.read(d,bbox)
        
        if loadData.shape[0] == 0:
            print('Empty File - Skipping')
            if d != fileList[-1]:
                continue
        
        #Hold the data for appending next files (if less than minimum record length)
        allData = loadData
        
        #Deterime if data being held (if less than minimum record length), if so concatenate
        if hold:
            if loadData.shape[0] == 0:
                allData = holdData
            else:
                allData = pd.concat([holdData,loadData])
        
        #Less than minimum record length so hold data and continue loop to append next files
        if allData.shape[0] < recordMin and d != fileList[-1]:
            hold = True
            holdData = allData
            continue
        else:
            hold = False
            holdData = None
            
        if allData.shape[0] == 0:
            continue
        
        #Must now be above minimum record length (or at end of file list)
        
        #Convert to geo coordinates and project to polar stereographic
        allData = GeoWrapper.convertToGeo(allData,cfg['data']['projection'],False)
        allData = GeoWrapper.project(allData,cfg['grid']['projection'])
        allData= GeoWrapper.extractXYtoCols(allData,0)
        allData = allData.drop(['geometry'],axis=1)
        
        #Create dataframe
        allData = pd.DataFrame(allData)
        
        #Write counters
        i = 0
        dLength = allData.shape[0]
        j = 0
        
        #Loop over data to create files of maximum record length
        while i <= dLength:
            
            increment = recordLength
            if i+recordLength+recordBuffer > dLength:
                increment = recordLength+recordBuffer
            
            #Take slice of data up to maximum data length
            data = allData[i:i+increment]
            data = data.reset_index().drop(['index'],axis=1)
        
            #Only do next steps if have data
            if data.shape[0]>0 :                      
                
                #Create index
                indexData = DataStoreHelper.createIndex(data,['lat','lon','x','y','startTime'])
                
                #Create files name
                fileTime = DataStoreHelper.createFileDateTime(indexData)
                fullPath = storeFolder + filePrefix + fileTime + '_' +str(j) + '.h5'
                
                #Write data
                store = pd.HDFStore(fullPath,mode='w',complevel=9, complib='blosc')
                store.append('data',data,index=False,data_columns=True)
                store.append('index',indexData,index=False,data_columns=True)
                store.close()
                
                #remove in-memory data to keep efficient 
                del data
                del indexData
            
            i += increment
            j += 1
            
        #remove in-memory data to keep efficient    
        del loadData
        del allData
        del holdData
        
        #Set if want to run a test
        if test:
            if ii >= 1:
                print("Time Taken: {}".format(time.time()-start))
                return
        
    print("Complete") 
    print("Time Taken: {}".format(time.time()-start))

#Done to ensure compatibility with dhdt code.
if __name__ == '__main__':
    main()
    
