import torch
import time
import pandas as pd

fname = '/media/martin/DATA/Data/hdf/areas/jakobshavn.h5'
baseOib = pd.read_hdf(fname,key="oib")
baseSwath = pd.read_hdf(fname,key="swath")






