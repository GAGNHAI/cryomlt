#Must have run joinData/py first

import pandas as pd

import matplotlib.pyplot as plt

compare = join[['WGS84EllipsoidHeight','elev_swath','elev_poca','distance']]

compare['HeightDiff'] = compare['WGS84EllipsoidHeight'] - compare['elev_swath']

#print('All data - Oib less Swath heights')
#compare.plot(x='distance',y='HeightDiff')

c500 = compare[compare['distance']<=500]
print('Distance less than 500m - Oib less Swath heights')
c500.plot.scatter(x='distance',y='HeightDiff')
plt.show()

c50 = compare[compare['distance']<=50]
print('Distance less than 50m - Oib less Swath heights')
c50.plot.scatter(x='distance',y='HeightDiff')
plt.show()

h100 = c50[abs(c50['HeightDiff'])<=500]
print('Distance less than 50m and Height diff less than 500 -  count: {}'.format(h100.shape[0]))

h100.plot.scatter(x='distance',y='HeightDiff')
plt.show()

#compare['HeightDiff']<
