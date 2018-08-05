#!/usr/bin/env python2

__all__ = ['GeoWrapper']

#Libaries
import geopandas as gpd
from shapely.geometry import Point
import logging
import time

class GeoWrapper(object):
    ''' Collection of generic static methods that help apply geopandas functions '''
        
    @staticmethod
    def convertToGeo(data,crs,dropLatLon):
        ''' Convert a pandas dataframe to a geopandas dataframe '''
        d = data
        if d is None:
            logging.debug('no data')
            return
        tstart = time.time()
        
        #Create geometry point
        geometry = [Point(xy) for xy in zip(d.lon, d.lat)]
        logging.debug('constructing list of points took %f seconds',time.time()-tstart)
        tstart = time.time()
        
        if dropLatLon:
            gd = gpd.GeoDataFrame(d.drop(['lon','lat'],axis=1),crs=crs,geometry=geometry)
        else:
            gd = gpd.GeoDataFrame(d,crs=crs,geometry=geometry)
        logging.debug('assembling geoframe took %f seconds',time.time()-tstart)
        return gd
       
    @staticmethod
    def project(data,crs):
        ''' Convert wgs86 to projected coordinates (crs) '''
        return data.to_crs(crs)
    
    @staticmethod
    def extractXYtoCols(data,precision=None):
        ''' Extracts x/y coordinates to two columns from a geometry point'''
        if precision == 0:
            data['x'] = data['geometry'].x.astype(int)
            data['y'] = data['geometry'].y.astype(int)
        else:
            data['x'] = data['geometry'].x
            data['y'] = data['geometry'].y
            if precision != None:
                data = data.round({'x': precision, 'y': precision})
        
        return data
        
        