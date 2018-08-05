#!/usr/bin/env python2

''' Convert Predicted data to a Shp file

Instructions:
    1) Set configuration section below
    2) Run
    
'''

#Libraries
from osgeo import ogr
import osgeo.osr as osr
import pandas as pd
import numpy as np
import os

###### Config ##########

folder = '/media/martin/FastData/Data/hdf/predictions/plots/HSmallRun_NN_L1_Adamax_50000_ScaleY/'
trainArea = 'all11to14'
testArea = 'jak15'
filOrFull = 'Full'

###### Start of Code #########

#Create shapefile driver
driver = ogr.GetDriverByName("ESRI Shapefile")

#Make output path
path = "{}{}/ShpFile_{}_{}/".format(folder,trainArea,testArea,filOrFull)
if not os.path.exists(path):
    os.makedirs(path)

#Create Shp file
data_source = driver.CreateDataSource(path + "{}_{}.shp".format(testArea,filOrFull))

#Set Projection
srs = osr.SpatialReference()
srs.ImportFromEPSG(3413)

#Create Layer
layer = data_source.CreateLayer("elevpoints", srs, ogr.wkbPoint)

#Add fields
field_name = ogr.FieldDefn("Name", ogr.OFTString)
field_name.SetWidth(24)
layer.CreateField(field_name)
field_region = ogr.FieldDefn("Region", ogr.OFTString)
field_region.SetWidth(24)
layer.CreateField(field_region)
layer.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Elev_Oib", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Elev_Swath", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Elev_Pred", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Diff_Pred", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Diff_Swath", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Diff_PvS", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("PowerScaled", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Coh_Swath", ogr.OFTReal))

#Load data
fname = "{}/{}/{}_{}.h5".format(folder,trainArea,testArea,filOrFull)
data= pd.read_hdf(fname,key="data")

#Drop Index
data2 = data.reset_index(drop=True)#[0:1000]

#Loop through predicted data and add to shp file
for i in np.arange(0,data2.shape[0]):
  print(i)
  #Create feature
  feature = ogr.Feature(layer.GetLayerDefn())
  
  #Set attributes
  feature.SetField("Name", 'Jak')
  feature.SetField("Region", 'Jak')
  feature.SetField("Latitude", data2['Y_Swath'][i])
  feature.SetField("Longitude", data2['X_Swath'][i])
  feature.SetField("Elev_Oib", data2['Elev_Oib'][i])
  feature.SetField("Elev_Swath", data2['Elev_Swath'][i])
  feature.SetField("Elev_Pred", data2['Predicted'][i])
  feature.SetField("Diff_Pred", data2['DiffPredicted'][i])
  feature.SetField("Diff_Swath", data2['DiffSwath'][i])
  feature.SetField("Diff_PvS", np.abs(data2['DiffPredicted'][i]) - np.abs(data2['DiffSwath'][i]))
  feature.SetField("PowerScaled", data2['PowerScaled_Swath'][i])
  feature.SetField("Coh_Swath", data2['Coh_Swath'][i])

  #Create the WKT for the point
  wkt = "POINT(%f %f)" %  (float(data2['X_Swath'][i]) , float(data2['Y_Swath'][i]))
  point = ogr.CreateGeometryFromWkt(wkt)

  #Set feature geometry
  feature.SetGeometry(point)
  
  #Create the feature in shp file
  layer.CreateFeature(feature)
  
  #Dereference feature
  feature = None

#Close data source
data_source = None