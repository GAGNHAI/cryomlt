import numpy as np
import argparse
import time
import dhdt
from dhdt.geoWrapper import GeoWrapper
import geopandas as gpd

from shapely.geometry import Point
from shapely.ops import nearest_points

#gpd1 = gpd.GeoDataFrame([['John',1,Point(1,1)],['Smith',1,Point(2,2)],['Soap',1,Point(0,2)]],columns=['Name','ID','geometry'])
#gpd2 = gpd.GeoDataFrame([['Work',Point(0,1.1)],['Shops',Point(2.5,2)],['Home',Point(1,1.1)]],columns=['Place','geometry'])
#pts3 = gpd2.geometry.unary_union    

#def near(point, pts=pts3):
     # find the nearest point and return the corresponding Place value
     #nearest = gpd2.geometry == nearest_points(point, pts)[1]
     #return gpd2[nearest].Place.get_values()[0]

#gpd1['Nearest'] = gpd1.apply(lambda row: near(row.geometry), axis=1)


#def main():
    
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


def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""
    # Find the geometry that is closest
    nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].get_values()[0]
    return value

baseWithId = baseSwath.reset_index()

chunkCS = 4000
chunkOIB = 3000

swath = GeoWrapper.convertToGeoXY(baseWithId[0:chunkCS],cfg['grid']['projection'])
oib = GeoWrapper.convertToGeoXY(baseOib[0:chunkOIB],cfg['grid']['projection'])

start = time.time()
    
unary_union = swath.unary_union

oib['nearest_id'] = oib.apply(nearest, geom_union=unary_union, df1=oib, df2=swath, src_column='index', axis=1)

taken = time.time() - start
print(taken)
print('Days to calc: {}'.format((18375400*801771/(chunkCS*chunkOIB))*taken/(60*60*24)))   
    
'''    
    i = 0
    oibLength = baseOib.shape[0]
    oibLength = 5
    
    minDists = np.array([[],[]])
    swath = GeoWrapper.convertToGeoXY(baseSwath[0:1000],cfg['grid']['projection'])
    oib = GeoWrapper.convertToGeoXY(baseOib[0:1000],cfg['grid']['projection'])
    
    gpd.sjoin.sjoin()
    
    allData = GeoWrapper.project(allData,cfg['grid']['projection'])
    
    while i <  oibLength:
    
        xMin = oib['x'][i]-1000
        xMax = oib['x'][i]+1000
        yMin = oib['y'][i]-1000
        yMax = oib['y'][i]+1000
        a = swath
        keep = (a['x']<=xMax) & (a['x']>=xMin) & (a['y']<=yMax) & (a['y']>=yMin)
        swathCut = a[keep]
        
        #baseX = ab['x'][0]
        #baseY = ab['y'][0]
        
        ones = np.ones(swathCut.shape[0])
        
        dists = (swathCut['x']-oib['x'][i]*ones)**2 + (swathCut['y']-oib['y'][i]*ones)**2
        
        minV = min(dists)
        ind = np.argmin(dists)
        minDists = np.append(minDists,[[minV],[ind]],axis=1)
        
        i += 1
        
    taken = time.time()-start

'''

#if __name__ == '__main__':
#    main()


