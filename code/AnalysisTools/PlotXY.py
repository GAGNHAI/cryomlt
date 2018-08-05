import matplotlib.pyplot as plt


data = j2011

data['swathMoib'] = data['Elev_Swath']-data['Elev_Oib']

data.plot.scatter(x='X_Swath',y='Y_Swath',c='Elev_Swath',s=1,cmap='YlOrRd')

data.plot.scatter(x='X_Swath',y='Y_Swath',c='swathMoib',s=1,cmap='RdYlGn',vmin=-1,vmax=1)

#comb = x_test
#comb['diffPred'] = y_predSet-y_testSet
#comb['diffSwath'] = x_test.Elev_Swath-y_testSet

#comb.plot.scatter(x='X_Swath',y='Y_Swath',c='diffSwath',s=1,cmap='RdYlGn',vmin=-1,vmax=1)
#comb.plot.scatter(x='X_Swath',y='Y_Swath',c='diffPred',s=1,cmap='RdYlGn',vmin=-1,vmax=1)