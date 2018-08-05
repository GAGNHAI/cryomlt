from osgeo import ogr
import osgeo.osr as osr
import pandas as pd
import numpy as np
import os

# set up the shapefile driver
driver = ogr.GetDriverByName("ESRI Shapefile")

folder = '/media/martin/FastData/Data/hdf/predictions/plots/HSmallRun_NN_L1_Adamax_50000_ScaleY/'
trainArea = 'all11to14'
testArea = 'jak15'
filOrFull = 'Full'

path = "{}{}/ShpFile_{}_{}/".format(folder,trainArea,testArea,filOrFull)
if not os.path.exists(path):
    os.makedirs(path)


# create the data source
data_source = driver.CreateDataSource(path + "{}_{}.shp".format(testArea,filOrFull))

srs = osr.SpatialReference()
srs.ImportFromEPSG(3413)

# create the layer
layer = data_source.CreateLayer("elevpoints", srs, ogr.wkbPoint)

# Add the fields we're interested in
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

fname = "{}/{}/{}_{}.h5".format(folder,trainArea,testArea,filOrFull)

#fname = '/media/martin/FastData/Data/hdf/predictions/jak11train_jak11test.h5'
data= pd.read_hdf(fname,key="data")

data2 = data.reset_index(drop=True)#[0:1000]

for i in np.arange(0,data2.shape[0]):
  print(i)
  # create the feature
  feature = ogr.Feature(layer.GetLayerDefn())
  # Set the attributes using the values from the delimited text file
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

  # create the WKT for the feature using Python string formatting
  wkt = "POINT(%f %f)" %  (float(data2['X_Swath'][i]) , float(data2['Y_Swath'][i]))

  # Create the point from the Well Known Txt
  point = ogr.CreateGeometryFromWkt(wkt)

  # Set the feature geometry using the point
  feature.SetGeometry(point)
  # Create the feature in the layer (shapefile)
  layer.CreateFeature(feature)
  # Dereference the feature
  feature = None

# Save and close the data source
data_source = None