__all__ = ['Config']

from configobj import ConfigObj, flatten_errors
from validate import Validator
import datetime
from collections import Mapping
import os.path

# the default configuration
defaultCfgStr = """
# miscellaneous configurations
[misc]
# the program can run in parallel. Use the nproc option to specify
# the number of processors that should be used in x and y direction
nproc = int_list(min=2,max=2,default=list(1,1))

# input data
[data]
# the data type, currently we only support L2Swath
type = option('L2Swath',default='L2Swath')
# the store type can be either hdf5 or a multifile dask store
storeType = option(hdf,dask,multihdf,default=hdf)
# dask temp directory
daskTmp = string(default=/tmp)
# the path to the intermediate pandas data store
store = string(default='store')
storePoca = string(default='storePoca')

#Option to load all available columns in data
allColumns = bool(default=False)

# the multihdf store needs a bounding box and cell sizes
store_bbox = float_list(min=4,max=4)
store_dlon = float(default=0.5)
store_dlat = float(default=0.5)

# the input glob
input = string(default='none')
inputPoca = string(default='none')

# definition of the input coordinate system
# the coordinate system uses proj4 parameters
[[projection]]
proj  = string(default=latlong)
ellps = string(default=WGS84)
datum = string(default=WGS84)
__many__ = float()

# time configuration
# dates are of the format yyyy-mm-dd HH:MM:SS
# time spans can be in years, months or days - append the unit
# a, m or d as appropriate
[time]
# date and time of time origin. All times are relative to this date
timeOrigin = string(default="1990-1-1 0:0:0")

# the length of a year in days
yearLength = float(default=365.2422)

# the time period, use date format yyyy-mm-dd
period = list(min=1,max=2)

# produce a time series with time step in years, months or days
timeStep = string(default=None)

# the time resolution in years, months or days. Consider measurements 
# in the perod t+/-t_resolution for fitting data
timeResolution = string(default=1m)

# grid configuration
[grid]
# the basename of the output file
output = string()
# place tiles into a deep directory structure
deepstore = boolean(default=True)

# the lower left and upper right corner of the bounding box of the mesh.
bbox = float_list(min=4,max=4)

# name of a shapefile containing a mask
mask = string(default=None)
# the buffer around the mask in m
mask_buffer = float(default=20000)

# the node spacing in meters
posting = float(default=500)

# mesh registration
# can be either node for node registration or pixel for pixel registration
registration = option(node,pixel,default=node)

# definition of the output coordinate system
# the coordinate system uses proj4 parameters
[[projection]]
proj  = string(default=stere)
lat_ts  = string(default=70.0)
lat_0  = string(default=90.0)
lon_0  = string(default=-45.0)
ellps = string(default=WGS84)
datum = string(default=WGS84)
__many__ = float()

# dhdt options
[dhdt]
# the search radius in meters
radius = float(min=0,default=1000.)
# only fit data if there is more than minData data available
minData = integer(default=10)
# mark pixels where abs(rate)>rateMax as NaN
maxRate = float(default=1e6)
# power threshold below which data is not considered
minPower = float(default=1e4)
# demDiffMad threshold in meter
maxDemDiffMad = float(default=10.)
# demDiff threshold in meter
maxDemDiff = float(default=100.)
# satellite pass, can be either D for ascending or A for descending, select None for all
satellitePass = option('A','D',default=None)
# polynomial model for fit
model = option('linear',quadratic',default='linear')
# function of weight in fit
weight = option('power4','ones',default='power4')
# leading edge width parameter
lew_where = float(min=0,default=0)
lew_opt = option('none','model','fix',default=None)
lew_prop = float(default=0)
lew_timeCorr = float(min=0,default=0)
# power parameter
pow_where = float(min=0,default=0)
pow_opt = option('none','power','poca','powerAndPoca',default=None)
pow_timeCorr = float(min=0,default=0)
"""

# populate the default  config object which is used as a validator
dhdtDefaults = ConfigObj(defaultCfgStr.split('\n'),list_values=False,_inspec=True)
validator = Validator()

def getTime(tstring):
    if tstring is None:
        return None
    u = tstring[-1]
    if u=='y':
        u='a'
    assert u in ['a','m','d']
    t = int(tstring[:-1])
    return {'value':t,'units':u}

def getDateTime(tstring):
    if tstring is None:
        return None
    return datetime.datetime.strptime(tstring,"%Y-%m-%d %H:%M:%S")

class TimeSection(Mapping):
    def __init__(self,cfg):
        assert 'time' in cfg.keys()
        self._cfg = cfg

    def __getitem__(self,item):
        v = self._cfg['time'][item]
        if item == 'period':
            t0 = getDateTime(v[0]+' 0:0:0')
            if len(v)>1:
                t1 = getDateTime(v[1]+' 0:0:0')
            else:
                t1 = None
            return t0,t1
        elif item == 'timeOrigin':
            return getDateTime(v)
        elif item in ['timeStep','timeResolution']:
            return getTime(v)
        else:
            return v

    def __len__(self):
        return len(self._cfg['time'])

    def __iter__(self):
        for k in self._cfg['time'].keys():
            yield k

class Config(Mapping):
    """object containing the dhdt configuration"""

    def __init__(self):
        self._cfg = ConfigObj(configspec=dhdtDefaults)
        self._cfg.validate(validator)
        self._path = ''

    def readCfg(self,fname):
        """read and parse configuration file"""

        self._cfg.filename = fname
        self._cfg.reload()
        self._path = os.path.dirname(fname)
        validated = self._cfg.validate(validator, preserve_errors=True)
        if validated is not True:
            # filter out store_bbox
            nerror = 0
            msg = 'Could not read config file {0}:'.format(fname)
            for s in validated:
                if validated[s] is not True:
                    msg += '\n in section {0}:'.format(s)
                    for v in validated[s]:
                        msg += '\n   {0}: {1}'.format(v,validated[s][v])
                        if not validated[s][v]:
                            nerror += 1

            if not validated['data']['store_bbox'] and self._cfg['data']['storeType'] != 'multihdf' and nerror == 1:
                return
            raise RuntimeError, msg

    def full_path(self,fname):
        if fname is None:
            return
        if not os.path.isabs(fname):
            return os.path.normpath(os.path.join(self._path,fname))
        
    def __getitem__(self,item):
        if item == 'time':
            return TimeSection(self._cfg)
        else:
            return self._cfg[item]

    def __iter__(self):
        for k in self._cfg.keys():
            yield k

    def __len__(self):
        return len(self._cfg)
    

        
if __name__ == '__main__':
    import sys

    cfg = Config()
    if len(sys.argv)>1:
        cfg.readCfg(sys.argv[1])

    for k in cfg:
        print k
        for kk in cfg[k]:
            print '  ',kk,cfg[k][kk]
    
