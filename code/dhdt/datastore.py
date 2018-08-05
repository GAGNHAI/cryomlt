__all__ = ['checkStore','getStore','getStoreName']

import pandas as pd
import logging
import time
import os.path
from datetime import datetime
import glob
import sys
from dataReader import *
from grid import GridLonLat
import math
import json
from geoWrapper import GeoWrapper

import dask.dataframe as dd
from dask.delayed import delayed
import dask

class DataStore(object):
    def __init__(self,dname,reader=None,mode='r',timeOrigin="1990-1-1 0:0:0"):
        self._dname = dname
        self._mode = mode
        self._timeOrigin = datetime.strptime(timeOrigin,"%Y-%m-%d %H:%M:%S")
        if reader is not None:
            self._dataReader = reader(timeOrigin=timeOrigin)
        else:
            self._dataReader = None

    @property
    def dname(self):
        return self._dname
            
    @property
    def mode(self):
        return self._mode

    @property
    def reader(self):
        return self._dataReader
    
    @property
    def timeOrigin(self):
        return self._timeOrigin

    @property
    def bbox(self):
        return self.getBBox()
    def getBBox(self):
        raise NotImplementedError

    @property
    def columns(self):
        return self.getColumns()
    def getColumns(self):
        raise NotImplementedError
    
    def time(self,time,fmt="%Y-%m-%d %H:%M:%S"):
        if isinstance(time,datetime):
            t=(time-self.timeOrigin).total_seconds()
        else:
            t=(datetime.strptime(time,fmt)-self.timeOrigin).total_seconds()
        return t

    def readData(self,fileglob):
        raise NotImplementedError
    
    def data(self,bbox=None,minPower=1e4,demDiffMadThresh=10.,demDiffThresh=100.,spass=None):
        raise NotImplementedError
  
    def getGeoPandas(self,crs=None,bbox=None,minPower=0,demDiffMadThresh=100.,demDiffThresh=1000.,spass=None):
        d = self.data(bbox=bbox,minPower=minPower,demDiffMadThresh=demDiffMadThresh,demDiffThresh=demDiffThresh,spass=spass)
        gd = GeoWrapper.convertToGeo(d,crs,True)
        return gd

class HDFDataStore(DataStore):
    def __init__(self,dname,reader=None,mode='r',timeOrigin="1990-1-1 0:0:0",cell=-1):
        DataStore.__init__(self,dname,reader=reader,mode=mode,timeOrigin=timeOrigin)
        logging.info('using hdf store %s'%self.dname)
        self._store = pd.HDFStore(self.dname,mode=mode,complevel=9, complib='blosc')

    def __del__(self):
        logging.debug('closing hdf5 store')
        self._store.close()

    def getBBox(self):
        tstart = time.time()
        bbox =  [self._store.data['lon'].min(),self._store.data['lat'].min(),
                 self._store.data['lon'].max(),self._store.data['lat'].max()]
        logging.debug('extracting bounding box took %f seconds',time.time()-tstart)
        return bbox

    def getColumns(self):
        tstart = time.time()
        row = self._store.select('data',start=0,stop=1)
        logging.debug('extracting columns took %f seconds',time.time()-tstart)
        return row.columns

    def data(self,bbox=None,minPower=1e4,demDiffMadThresh=10.,demDiffThresh=100.,spass=None):
        constraints = []
        # FIXME: need to figure out how to deal with wrapping longitude boundary
        if bbox!=None:
            for i in [0,2]:
                if bbox[i] > 180:
                    bbox[i] -= 360.
            constraints += ['lon>=%f'%bbox[0],
                            'lon<=%f'%bbox[2],
                            'lat>=%f'%bbox[1],
                            'lat<=%f'%bbox[3]]

        if 'power' in self.columns:
            constraints.append('power>%f'%minPower)
        if 'demDiffMad' in self.columns:
            constraints.append('demDiffMad<%f'%demDiffMadThresh)
        if 'demDiff' in self.columns:
            constraints.append('demDiff<%f'%demDiffThresh)
        if spass is not None and 'heading' in self.columns:
            if spass == 'D':
                # ascending passes heading ~> -10deg
                constraints.append('heading>0')
            elif spass == 'A':
                # descending passes heading ~> 190deg
                constraints.append('heading<0')
            else:
                raise ValueError, 'wrong value for spass'
        

        tstart = time.time()
        data = self._store.select('data',where=constraints)
        logging.debug('selecting data took %f seconds',time.time()-tstart)
        return data
    
    def readData(self,fileglob):
        for d in glob.glob(fileglob):
            self._store.append('data',self.reader.read(d),index=False,data_columns=True)




class MultiHDFDataStore(DataStore):
    def __init__(self,dname,reader=None,mode='r',timeOrigin="1990-1-1 0:0:0",bbox=None,dlon=0.5,dlat=0.5,cell=-1):
        DataStore.__init__(self,dname,reader=reader,mode=mode,timeOrigin=timeOrigin)
        logging.info('using multi hdf store %s'%self.dname)
        
        self._cell = cell

        # read the meta data from the store
        with open(os.path.join(self.dname,'meta.json'),'r') as mfile:
            m = json.load(mfile)
            self._grid = GridLonLat(m['bbox'],dlon=m['dlon'],dlat=m['dlat'])
        
        
        if bbox is not None:
            tmpGrid = GridLonLat(bbox,dlon=dlon,dlat=dlat)
            match = True
            for i in range(4):
                if abs(self._grid.bbox[i] - tmpGrid.bbox[i]) > 1e-6:
                    match = False
            if (self._grid.dlon - tmpGrid.dlon) > 1.e-6:
                match = False
            if (self._grid.dlat - tmpGrid.dlat) > 1.e-6:
                match = False
            if not match:
                logging.error('lon/lat grid do not match')
                sys.exit(1)
            
        if mode == 'w':
            self._store = pd.HDFStore(self.storeName,mode='w',complevel=9, complib='blosc')
        else:
            self._store = None

    def __del__(self):
        if self._store is not None:
            logging.debug('closing hdf5 store')
            self._store.close()            
            
    def getBBox(self,n=None):
        return self._grid.getBBox(n=n)
            
    def getStoreName(self,cell):
        assert cell >= 0 and cell < self._grid.ncells
        path = '{:x}'.format(cell)
        out = os.path.join(self.dname,'multihdf')
        for p in path[:-1]:
            out = os.path.join(out,p)
        return os.path.join(out,'%06d.h5'%cell)
    @property
    def storeName(self):
        return self.getStoreName(self._cell)

    def getColumns(self):
        tstart = time.time()
        rows = None
        for i in range(self._grid.ncells):
            haveCols = True
            logging.debug('opening store %s'%self.getStoreName(cell=i))
            try:
                store = pd.HDFStore(self.getStoreName(cell=i),mode='r',complevel=9, complib='blosc')
                row = store.select('data',start=0,stop=1)
                rows = row.columns
            except:
                haveCols = False
            store.close()
            logging.debug('closing store %s'%self.getStoreName(cell=i))
            if haveCols:
                break
        logging.debug('extracting columns took %f seconds',time.time()-tstart)
        return rows

    def data(self,bbox=None,minPower=1e4,demDiffMadThresh=10.,demDiffThresh=100.,spass=None):
        constraints = []
        # FIXME: need to figure out how to deal with wrapping longitude boundary
        if bbox!=None:
            for i in [0,2]:
                if bbox[i] > 180:
                    bbox[i] -= 360.
            constraints += ['lon>=%f'%bbox[0],
                            'lon<=%f'%bbox[2],
                            'lat>=%f'%bbox[1],
                            'lat<=%f'%bbox[3]]

        if 'power' in self.columns:
            constraints.append('power>%f'%minPower)
        if 'demDiffMad' in self.columns:
            constraints.append('demDiffMad<%f'%demDiffMadThresh)
        if 'demDiff' in self.columns:
            constraints.append('demDiff<%f'%demDiffThresh)
        if spass is not None and 'heading' in self.columns:
            if spass == 'D':
                # ascending passes heading ~> -10deg
                constraints.append('heading>0')
            elif spass == 'A':
                # descending passes heading ~> 190deg
                constraints.append('heading<0')
            else:
                raise ValueError, 'wrong value for spass'
        

        tstart = time.time()
        data = None
        
        for cell in self._grid.getCells(bbox):
            logging.debug('opening store %s'%self.getStoreName(cell=cell))
            store = pd.HDFStore(self.getStoreName(cell=cell),mode='r',complevel=9, complib='blosc')
            try:
                d = store.select('data',where=constraints)
            except:
                logging.debug('no data in hdf file')
                continue
            if data is None:
                data = d
            else:
                data = data.append(d,ignore_index=True)
            store.close()
            logging.debug('closing store %s'%self.getStoreName(cell=cell))
        logging.debug('selecting data took %f seconds',time.time()-tstart)
        return data
    
    def readData(self,fileglob):
        bbox = self.getBBox(n=self._cell)
        for d in glob.glob(fileglob):
            data = self.reader.read(d)
            self._store.append('data',data[(data.lon>=bbox[0]) & (data.lon<bbox[2]) & (data.lat>=bbox[1]) & (data.lat<bbox[3])],
                               index=False,data_columns=True)

    
class DaskDataStore(DataStore):
    def __init__(self,dname,reader=None,mode='r',timeOrigin="1990-1-1 0:0:0",dlon=0.5,dlat=0.5,cell=-1):
        DataStore.__init__(self,dname,reader=reader,mode=mode,timeOrigin=timeOrigin)

        logging.info('using dask store %s'%self.dname)
        
        self._metaFields = ['dlon','dlat','bbox','spatialFactor','ni','nj']
        self._daskPattern = os.path.join(self.dname,'dstore*.h5')
        if mode == 'w':
            self._store = None

            for m in self._metaFields:
                setattr(self,'_'+m,None)        
            self._dlon = dlon
            self._dlat = dlat
        else:
            self._store = dd.read_hdf(self._daskPattern,'/data')
            self._loadMetaData()
            
    @property
    def dlon(self):
        return self._dlon
    @property
    def dlat(self):
        return self._dlat
    @property
    def ni(self):
        return self._ni
    @property
    def nj(self):
        return self._nj
    @property
    def spatialFactor(self):
        return self._spatialFactor

    def _storeMetaData(self):
        if not os.path.exists(self.dname):
            os.makedirs(self.dname)
        else:
            assert os.path.isdir(self.dname)
        
        meta = {}
        for m in self._metaFields:
            meta[m] = getattr(self,m)

        with open(os.path.join(self.dname,'meta'),'w') as mfile:
            json.dump(meta,mfile)

    def _loadMetaData(self):                
        with open(os.path.join(self.dname,'meta'),'r') as mfile:
            meta = json.load(mfile)
            for m in self._metaFields:
                setattr(self,'_'+m,meta[m])
            
    def getBBox(self):
        return self._bbox

    def getColumns(self):
        return self._store.columns
    
    def computeIdx(self,lon,lat,t):
        i = (lon-self.bbox[0])//self.dlon
        j = (lat-self.bbox[1])//self.dlat
        idx = self.spatialFactor*(i+j*self.ni)+t
        return idx
    
    def readData(self,fileglob):
        dfs = [delayed(self.reader.read)(fn) for fn in glob.glob(fileglob)]
        self._store = dd.from_delayed(dfs)

        # min/max lon/lat
        tstart = time.time()
        dmin = self._store[['lon','lat','time']].min().compute()
        dmax = self._store[['lon','lat','time']].max().compute()
        logging.info('getting bounding box took %f seconds',time.time()-tstart)
        ll = (dmin['lon']//self.dlon)*self.dlon,(dmin['lat']//self.dlat)*self.dlat
        ur = math.ceil(dmax['lon']/self.dlon)*self.dlon,math.ceil(dmax['lat']/self.dlat)*self.dlat
        
        self._spatialFactor = 10**(math.ceil(math.log10(dmax['time'])+1))
        self._bbox = [ll[0],ll[1],ur[0],ur[1]]
        self._ni = int((ur[0]-ll[0])/self.dlon)
        self._nj = int((ur[1]-ll[1])/self.dlat)

        # store meta data
        self._storeMetaData()
        
        self._store['idx'] = self.computeIdx(self._store['lon'],self._store['lat'],self._store['time'])
        self._store.set_index('idx')

        self._store = self._store.repartition(npartitions=self._store.npartitions//10)

        tstart = time.time()
        dd.to_hdf(self._store,self._daskPattern,'/data')
        os.utime(self.dname,None)
        logging.info('storing dataset took %f seconds',time.time()-tstart)

    def data(self,bbox=None,minPower=1e4,demDiffMadThresh=10.,demDiffThresh=100.,spass=None):

        tstart = time.time()
        #data = pd.DataFrame()
        # 
        #nlat = int((bbox[3]-bbox[1])//self.dlat)+1
        #for i in range(nlat):
        #    slat = bbox[1]+i*self.dlat
        #    elat = min(bbox[3],bbox[1]+(i+1)*self.dlat)
        #    sidx = self.computeIdx(bbox[0],slat,0)
        #    eidx = self.computeIdx(bbox[2]+self.dlon,elat,0)
        #    
        #    d=self._store[(self._store.lat>=bbox[1]) & (self._store.lat<=bbox[3]) &
        #                  (self._store.lon>=bbox[0]) & (self._store.lon<=bbox[2])].compute()


            
        #     data = data.append(d)
        
        data =self._store[(self._store.lat>=bbox[1]) & (self._store.lat<=bbox[3]) &
                          (self._store.lon>=bbox[0]) & (self._store.lon<=bbox[2])]
        if 'power' in self.columns:
            data = data[data.power>minPower]
        if 'demDiffMad' in self.columns:
            data = data[data.demDiffMad<demDiffMadThresh]
        if 'demDiff' in self.columns:
            data = data[data.demDiff<demDiffThresh]
        if spass is not None and 'heading' in self.columns:
            if spass == 'D':
                # ascending passes heading ~> -10deg
                data = data[data.heading>0]
            elif spass == 'A':
                # descending passes heading ~> 190deg
                data = data[data.heading<0]
            else:
                raise ValueError, 'wrong value for spass'
        data = data.compute()
        logging.debug('selecting data took %f seconds',time.time()-tstart)
        return data
        
def checkStore(cfg):
    """return True if store is uptodate"""

    if not os.path.exists(cfg['data']['store']):
        return False
    store_mtime =  os.path.getmtime(cfg['data']['store'])
    for d in glob.glob(cfg['data']['input']):
        if os.path.getmtime(d) > store_mtime:
            return False
    return True

def getStoreName(cfg):
    sname = None
    if cfg['data']['storePoca'] is not None:
        sname = cfg['data']['storePoca']
    if cfg['data']['store'] is not None:
        sname = cfg['data']['store']
    if sname is None:
        raise RuntimeError, 'no store name specified'
    return sname

def getStore(cfg,mode='r',cell=-1):
    data = None
    dataPoca = None

    if cfg['data']['storeType'] == 'hdf':
        Store = HDFDataStore
    elif cfg['data']['storeType'] == 'dask':
        Store = DaskDataStore
        if mode=='w':
            dask.set_options(get=dask.multiprocessing.get)
        dask.set_options(temporary_directory=cfg['data']['daskTmp'])
    elif cfg['data']['storeType'] == 'multihdf':
        Store = MultiHDFDataStore
    else:
        raise ValueError, 'Unkown store type %s'%cfg['data']['storeType'] 
    
    if mode == 'r':    
        if cfg['data']['input'] != 'none':
            data = Store(cfg['data']['store'],mode='r')
        if cfg['data']['inputPoca'] != 'none':
            dataPoca = Store(cfg['data']['storePoca'],mode='r')
    elif mode == 'w':
        allCols = False
        if 'allColumns' in cfg['data']:
            allCols = cfg['data']['allColumns']
        if cfg['data']['input'] != 'none':
            if allCols:
                data = Store(cfg['data']['store'],reader=L2SwathReaderAll,mode='w',cell=cell)
            else:
                data = Store(cfg['data']['store'],reader=L2SwathReader,mode='w',cell=cell)
        if cfg['data']['inputPoca'] != 'none':
            if allCols:
                dataPoca = Store(cfg['data']['storePoca'],reader=L2SwathReaderAll,mode='w',cell=cell)
            else:
                dataPoca = Store(cfg['data']['storePoca'],reader=L2SwathReader,mode='w',cell=cell)
    else:
        raise ValueError, 'mode must be either r or w'

    return data,dataPoca
