__all__ = ['readData']

# Run with a config - e.g. /media/martin/FastData/Code/cryomlt/configs/testBoth.cfg

import dhdt
import argparse
import logging
import time
import glob
from dhdt.geoWrapper import GeoWrapper
from dhdt.datastoreHelper import DataStoreHelper
import pandas as pd

def main():
    
    ######### Load Config ###########
    
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


    start = time.time()

    ########## Config - need to add to config file ####
     
    #Jakobshavn
    #bbox = [-51.3, 68.2, -47.5, 70.3]
    
    #Strostrommen
    bbox = [-32.8, 77.6, -14.8, 79.5]
    
    midLon = (bbox[0]+bbox[2])/2
    midLat = (bbox[1]+bbox[3])/2

    ############# Start of code ##############
    
    data = pd.DataFrame({'corner': ['LowerLeft','UpperRight','LowLonMidLat','MidLonLowLat','HighLonMidLat','MidLonHighLat'],
                         'lon':[bbox[0],bbox[2],bbox[0],midLon,bbox[2],midLon], 
                         'lat':[bbox[1],bbox[3],midLat,bbox[1],midLat,bbox[3]]})

    allData = GeoWrapper.convertToGeo(data,cfg['data']['projection'],False)
    allData = GeoWrapper.project(allData,cfg['grid']['projection'])
    allData = GeoWrapper.extractXYtoCols(allData,0)
    
    print(allData)
    
    xyBBox = [min(allData['x']),min(allData['y']),max(allData['x']),max(allData['y'])]

    print("Transformed BBox")
    print(xyBBox)
    
    print("Width: {:.2f}km, Length: {:.2f}km".format((xyBBox[2]-xyBBox[0])/1000.0,(xyBBox[3]-xyBBox[1])/1000.0))
      
    print("Complete")
    
    print(time.time()-start)

if __name__ == '__main__':
    main()
    
