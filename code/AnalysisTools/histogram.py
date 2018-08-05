import pandas as pd
import matplotlib.pyplot as plt


fname = '/media/martin/FastData/Data/hdf/areas/jakobshavn2011.h5'
data = pd.read_hdf(fname,key="swath")
dataP = pd.read_hdf(fname,key="poca")

plt.hist(data['demDiff'],bins=101,range=[-500,500])

