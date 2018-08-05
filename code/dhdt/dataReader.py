__all__ = ['DataReader','L2SwathReader','L2SwathReaderAll','MatlabReader','CSVReader']

import numpy
import time
import logging
from datetime import datetime
import os.path
import scipy.io as sio
import pandas as pd
import math

def nans(shape,dtype=float):
    a = numpy.empty(shape, dtype)
    a.fill(numpy.nan)
    return a

class DataReader(object):
    def __init__(self,timeOrigin="1990-1-1 0:0:0"):
       self._timeOrigin = datetime.strptime(timeOrigin,"%Y-%m-%d %H:%M:%S")

    @property
    def timeOrigin(self):
        return self._timeOrigin

    def time(self,time,fmt="%Y-%m-%d %H:%M:%S"):
        if isinstance(time,datetime):
            t=(time-self.timeOrigin).total_seconds()
        else:
            t=(datetime.strptime(time,fmt)-self.timeOrigin).total_seconds()
        return t
    
class L2SwathReader(DataReader):
    def __init__(self,timeOrigin="1990-1-1 0:0:0"):
        DataReader.__init__(self,timeOrigin=timeOrigin)

        self._cols = ['lon','lat','elev','coh','phase','power','demDiff','demDiffMad','meanDiffSpread','wf_number','phaseAmb','heading','oceanTide','leadEdgeW','poca_power']
        self._mpTancCols = ['sizeSurf','stdAoA','mAoA','peakWf']
        self._dtypes = {}
        for c in self._cols+self._mpTancCols:
            self._dtypes[c] = numpy.float64

        self._dtypes['wf_number'] = numpy.uint16
        self._dtypes['phaseAmb'] = numpy.int16
        
        self._dtypes['time'] = numpy.float64
        self._dtypes['startTime'] = numpy.float64

    def read(self,fname):
        logging.debug('reading %s',fname)
        dFrame = pd.DataFrame()
        for c in self._cols + self._mpTancCols + ['time','startTime']:
            dFrame[c] = pd.Series(dtype=self._dtypes[c])
        try:
            lstart = time.time()
            start = self.time(os.path.basename(fname)[19:34],fmt="%Y%m%dT%H%M%S")
            stop = self.time(os.path.basename(fname)[35:50],fmt="%Y%m%dT%H%M%S")
            mat = sio.loadmat(fname)
            ndata = mat['lon'].shape[0]
            for c in self._cols:
                if c in mat.keys():
                    dFrame[c] = numpy.array(mat[c][:,0],dtype=self._dtypes[c])
                else:
                    dFrame[c] = nans((ndata,),dtype=self._dtypes[c])
            if 'MPTanc' in mat.keys():
                for mc in self._mpTancCols:
                    dFrame[mc] = numpy.array(mat['MPTanc'][mc][0][0][:,0],dtype=self._dtypes[c])
            else:
                for mc in self._mpTancCols:
                    dFrame[mc] = nans((ndata,),dtype=self._dtypes[c])

            # make sure longitude is -180<lon<180
            dFrame.loc[dFrame.lon > 180, 'lon'] -= 360.

            dFrame['time'] = numpy.linspace(start,stop,dFrame['lon'].shape[0])
            dFrame['startTime'] = start*numpy.ones(dFrame['lon'].shape[0])

            logging.debug("reading file %s took %f seconds",fname,time.time()-lstart)
        except:
            logging.error("failed to read file %s",fname)
        return dFrame

class L2SwathReaderAll(DataReader):
    def __init__(self,timeOrigin="1990-1-1 0:0:0"):
        DataReader.__init__(self,timeOrigin=timeOrigin)

        self._cols = []
        self._dtypes = {}

        self._dtypes['wf_number'] = numpy.uint16
        self._dtypes['phaseAmb'] = numpy.int16
        
        self._dtypes['time'] = numpy.float64
        self._dtypes['startTime'] = numpy.float64

    def read(self,fname):
        logging.debug('reading %s',fname)
        dFrame = pd.DataFrame()
        try:
            #Need to reset col list on each load
            self._cols = []
            
            mat = sio.loadmat(fname)
            ndata = mat['lon'].shape[0]
            for c in mat.keys():
                if type(mat[c]) == numpy.ndarray:
                    if mat[c].shape[0] == ndata:
                        self._cols.append(c)
                        if not c in self._dtypes:
                            self._dtypes[c] = numpy.float64
            for c in self._cols + ['time','startTime']:
                dFrame[c] = pd.Series(dtype=self._dtypes[c])
             
            lstart = time.time()
            start = self.time(os.path.basename(fname)[19:34],fmt="%Y%m%dT%H%M%S")
            stop = self.time(os.path.basename(fname)[35:50],fmt="%Y%m%dT%H%M%S")
            for c in self._cols:
                if c in mat.keys():
                    dFrame[c] = numpy.array(mat[c][:,0],dtype=self._dtypes[c])
                else:
                    dFrame[c] = nans((ndata,),dtype=self._dtypes[c])

            # make sure longitude is -180<lon<180
            dFrame.loc[dFrame.lon > 180, 'lon'] -= 360.

            dFrame['time'] = numpy.linspace(start,stop,dFrame['lon'].shape[0])
            dFrame['startTime'] = start*numpy.ones(dFrame['lon'].shape[0])

            logging.debug("reading file %s took %f seconds",fname,time.time()-lstart)
        except:
            logging.error("failed to read file %s",fname)
        return dFrame

class MatlabReader(DataReader):
    ''' Reads Swath and Poca Matlab files and converts to dataframes '''
    
    def __init__(self,timeOrigin="1970-1-1 0:0:0"):
        DataReader.__init__(self,timeOrigin=timeOrigin)

        #Set predefined column types
        self._cols = []
        self._dtypes = {}
        self._dtypes['wf_number'] = numpy.uint16
        self._dtypes['sampleNb'] = numpy.uint16
        self._dtypes['powerScaled'] = numpy.uint16
        self._dtypes['phaseAmb'] = numpy.int16
        

    def read(self,fname,bbox=None):
        logging.debug('reading %s',fname)
        dFrame = pd.DataFrame()
        try:
            
            self._cols = []
            
            #Load matlab data
            mat = sio.loadmat(fname)
            
            #Check not empty
            ndata = mat['lon'].shape[0]            
            if ndata == 0:
                return dFrame
            
            #Assign types
            for c in mat.keys():
                if type(mat[c]) == numpy.ndarray:
                    if mat[c].shape[0] == ndata:
                        self._cols.append(c)
                        if not c in self._dtypes:
                            self._dtypes[c] = numpy.float64
            for c in self._cols:
                dFrame[c] = pd.Series(dtype=self._dtypes[c])
             
            lstart = time.time()
            
            #Change longitude to -180 -> 180
            mat['lon'][mat['lon'][:,0]>180,0] -= 360
            
            #Get bounding box
            if bbox != None:
                inBox = (mat['lat'][:,0]>bbox[0]) & (mat['lon'][:,0]>bbox[1]) & (mat['lat'][:,0]<bbox[2]) & (mat['lon'][:,0]<bbox[3])
                #If all results are false, then return empty array, else return the part that is true
                if inBox[inBox].shape[0]>0:
                    for c in self._cols:
                        dFrame[c] = numpy.array(mat[c][inBox,0],dtype=self._dtypes[c])
                else:
                    for c in self._cols:
                        dFrame[c] = numpy.array([],dtype=self._dtypes[c])
            else:
                #Return everything as no bounding box requested
                for c in self._cols:
                    dFrame[c] = numpy.array(mat[c][:,0],dtype=self._dtypes[c])
            
            
            #Add time
            startTime = datetime.strptime(os.path.basename(fname)[19:34],"%Y%m%dT%H%M%S")
            startSecs = time.mktime(startTime.timetuple())
            dFrame['startTime'] = (startSecs*numpy.ones(dFrame['lon'].shape[0])).astype(int)
                       
            #Remove nulls
            notNullLatLon = dFrame['lon'].notnull() & dFrame['lat'].notnull()
            dFrameN = dFrame[notNullLatLon]
            
            #Clear memory
            del dFrame
            dFrame = dFrameN
            
            #DHDT Logging
            logging.debug("reading file %s took %f seconds",fname,time.time()-lstart)
        except:
            logging.error("failed to read file %s",fname)
        return dFrame
    
class CSVReader(DataReader):
    ''' Reads CSV file and converts to a dataframe '''
    
    def __init__(self,timeOrigin="1970-1-1 0:0:0"):
        DataReader.__init__(self,timeOrigin=timeOrigin)
       
    def read(self,fname,bbox=None):
        logging.debug('reading %s',fname)
        dFrame = pd.DataFrame()
        try:
            lstart = time.time()
            
            #Load CSV
            dFrame = pd.read_csv(fname,header=9,sep=', ',engine='python') #The file headers at at the 9th row
                       
            #Make lat, lon names consistent
            dFrame.rename(columns={"Latitude(deg)":"lat", "Longitude(deg)":"lon"},inplace=True)
            
            #If empty, return
            ndata = dFrame['lon'].shape[0]
            if ndata == 0:
                return dFrame
            
            # make sure longitude is -180<lon<180
            dFrame.loc[:]['lon'][dFrame['lon']>180] -= 360
            
            #Remove unwanted chars
            for col in dFrame.columns.values:
                newCol = col.replace("_", "")
                newCol = newCol.replace("-", "")
                newCol = newCol.replace("# ", "")
                newCol = newCol.replace("(m)", "")
                newCol = newCol.replace("(cm)", "")
                dFrame.rename(columns={col:newCol},inplace=True)
            
            #Remove unwanted cols
            dFrame = dFrame.drop(['UTCSecondsOfDay'],axis=1) 
           
            #Get bounding box
            if bbox != None:
                inBox = (dFrame['lat']>bbox[0]) & (dFrame['lon']>bbox[1]) & (dFrame['lat']<bbox[2]) & (dFrame['lon']<bbox[3])
                #If all results are false, then return empty array, else return the part that is true
                dFrame = dFrame[inBox]
            
            # Time code          
            startTime = datetime.strptime(os.path.basename(fname)[7:22],"%Y%m%d_%H%M%S")
            startSecs = time.mktime(startTime.timetuple())
            dFrame['startTime'] = (startSecs*numpy.ones(dFrame['lon'].shape[0])).astype(int)

            #Remove nulls
            notNullLatLon = dFrame['lon'].notnull() & dFrame['lat'].notnull()
            dFrameN = dFrame[notNullLatLon]
            del dFrame
            dFrame = dFrameN

            #DHDT Logging
            logging.debug("reading file %s took %f seconds",fname,time.time()-lstart)
        except:
            logging.error("failed to read file %s",fname)
        return dFrame
    
    
if __name__ == '__main__':
    import sys
    import dask.dataframe as dd
    from dask.delayed import delayed
    import dask

    l2 = L2SwathReader()

    dask.set_options(get=dask.multiprocessing.get)
    dask.set_options(temporary_directory='/scratch/local/mhagdorn/tt')

    dfs = [delayed(l2.read)(fn) for fn in sys.argv[1:]]
    df = dd.from_delayed(dfs)

    # get data bounding box
    dlon = 0.5
    dlat = 0.5
    dmin = df[['lon','lat','time']].min().compute()
    dmax = df[['lon','lat','time']].max().compute()

    spatialFactor = 10**(math.ceil(math.log10(dmax['time'])+1))
    
    ll = (dmin['lon']//dlon)*dlon,(dmin['lat']//dlat)*dlat
    ur = math.ceil(dmax['lon']/dlon)*dlon,math.ceil(dmax['lat']/dlat)*dlat
    ni = int((ur[0]-ll[0])/dlon)
    nj = int((ur[1]-ll[1])/dlat)

    print ll
    print ur
    print ni,nj

    i = (df['lon']-ll[0])//dlon
    j = (df['lat']-ll[1])//dlat

    df['idx'] = spatialFactor*(i+j*ni)+df['time']

    df = df.set_index('idx')

    df = df.repartition(npartitions=df.npartitions//10)
    
    dd.to_hdf(df,'/tmp/Ttest*.h5','/data')
    
    print df.npartitions


