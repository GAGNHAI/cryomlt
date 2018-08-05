# CryoMLT process Run Order

### 1) createSharedStores.py
- Converts matlab and csvn files to h5
- Shards data in to sensible chunks
- Creates index
- Saves a folder with the shards

### 2) createAreaStore.py
- loads swath, poca, oib
- filters on regions of interest
- creates single h5 store of area
- the data has not yet been joined

### 3) joinNearest.py or joinAll.py
- loads an area store from a h5 file
- performs nearest neighbour calc
- saves a distIndex to the h5 file

### 4) CreateTimeSeriesStore.py
- loads 2011 to 2016 data to single file

or load for a single area:
### 4b) loadJoinedData.py
- loads an area
- joins swath, poca, oib based on distIndex

### 5) prepareDataForML.py
- loads timeseries data for jak, st and se
- apply filter (coh>0.2, powerScaled>2500)
- remove before leading edge start
- save single file with all jak, st and se, 2011-16 data with keys (jak11,jak12,st11,se11 etc)

### 6) runMachineLearning.py
- load prepared data
- run NN, SVR or LR
- save results analysis
- save models

###7) loadAndApplyModel.py
- load saved model
- apply to other Cs2 data
- save resultant hdf5

### 8) generateShpFile.py
- load hdf5 file
- create and save shp file

or
### 8b) generateDemDhdt.py
- load 2011 and 2012 hdf5 files for single area
- create DEM and dhdts


# Folders

### dhdt
- modified version of library created by Magnus Hagdorn
- modified files: dataReader.py, dataStoreHelper.py. geoWrapper.py

### AnalysisTools
- collection of useful python files for adhoc analysis (not official)

### Visuals
- collection of useful python visualization files for adhoc analysis (not official)

### Other
- alternate attempts at methods (i.e. different join methods or NN models)





