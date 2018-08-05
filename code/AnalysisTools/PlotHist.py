import matplotlib.pyplot as plt
from scipy.stats import kde

#x=y_test
#y=nnResult.Y_pred

#nbins=300
#k = kde.gaussian_kde([x,y])
#xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
#zi = k(np.vstack([xi.flatten(), yi.flatten()]))


# Make the plot
#plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
#plt.colorbar()
#plt.show()
 

#plt.hist2d(y_test,nnResult.Y_pred, bins=300,range=[[500,2000],[500,2000]])
#plt.ylim(500,2000)
#plt.xlim(500,2000)
plt.hist2d(y_test,x_test['Elev_Swath'], bins=300)
plt.ylim(0,2000)
plt.xlim(0,2000)
plt.xlabel('Actual Elevation (m)')
plt.ylabel('Predicted Elevation (m)')
plt.title('Neural Network')
plt.colorbar()
plt.savefig("/media/martin/FastData/Models/Pytorch/SaveTest")
plt.show()
