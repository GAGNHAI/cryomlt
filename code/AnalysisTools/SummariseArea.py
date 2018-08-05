import pandas as pd

fname = '/media/martin/FastData/Data/hdf/areas/strostrommen2011.h5'
distInd = pd.read_hdf(fname,key="distIndex")

distUnq = distInd.loc[distInd.groupby("swathIndex")["distance"].idxmin()]

print('Total matched: {}'.format(distInd.shape[0]))
print('Points with 1000m: {}'.format(distInd[distInd['distance']<=1000].shape[0]))
print('Points with 500m: {}'.format(distInd[distInd['distance']<=500].shape[0]))
print('Points with 50m: {}'.format(distInd[distInd['distance']<=50].shape[0]))

print('Total Unique matched: {}'.format(distUnq.shape[0]))
print('Unique Points with 1000m: {}'.format(distUnq[distUnq['distance']<=1000].shape[0]))
print('Unique Points with 500m: {}'.format(distUnq[distUnq['distance']<=500].shape[0]))
print('Unique Points with 50m: {}'.format(distUnq[distUnq['distance']<=50].shape[0]))

#print('Total Unique/Poca matched: {}'.format(join.shape[0]))
#print('Unique/Poca Points with 1000m: {}'.format(join[join['distance']<=1000].shape[0]))
#print('Unique/Poca Points with 500m: {}'.format(join[join['distance']<=500].shape[0]))
#print('Unique/Poca Points with 50m: {}'.format(join[join['distance']<=50].shape[0]))