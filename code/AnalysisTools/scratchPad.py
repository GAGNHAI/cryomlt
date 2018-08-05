count = 0
for i in store.keys():
    count += store[i].shape[0]

print(count/2)

storeFolder = '/media/martin/FastData/Data/hdf/timeseries-nn/'

j = pd.HDFStore(storeFolder+storeJakName,mode='r',complevel=9, complib='blosc')
st = pd.HDFStore(storeFolder+storeStrName,mode='r',complevel=9, complib='blosc')
se = pd.HDFStore(storeFolder+storeSeName,mode='r',complevel=9, complib='blosc')

count2 = 0
for i in j.keys():
    count2 += j[i].shape[0]
for i in st.keys():
    count2 += st[i].shape[0]    
for i in se.keys():
    count2 += se[i].shape[0]    

print(count2)