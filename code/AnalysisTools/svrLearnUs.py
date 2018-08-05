import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

# #############################################################################
# Generate sample data
#X = np.sort(5 * np.random.rand(40), axis=0)
#X2 = np.sort(5 * np.random.rand(40), axis=0)
#X3 = np.sort(-8 * np.random.rand(40), axis=0)
#y = np.sin(X).ravel()+X2-np.cos(X3).ravel()



#X = x_train['Elev_Swath']
x = x_train#[['Elev_Swath','Coh_Swath']]
xtest = x_test#[['Elev_Swath','Coh_Swath']]
y = y_train

# #############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e7, gamma=1e-12, verbose=True)
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3, max_iter=100000)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)

#x = pd.DataFrame(data={'X':X})#,'X2':X2,'X3':X3})


y_rbf = svr_rbf.fit(x, y).predict(xtest)
#y_lin = svr_lin.fit(x, y).predict(x)
#y_poly = svr_poly.fit(x, y).predict(x)

# #############################################################################
# Look at the results
lw = 2
#plt.scatter(X, y, color='darkorange', label='data')
#plt.scatter(X2, y, color='blue', label='data')
plt.scatter(y_test, y_rbf, color='navy', lw=lw, label='RBF model')
#plt.scatter(y_test, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
'''
plt.scatter(x_test['Coh_Swath'], y_rbf, color='navy', lw=lw, label='RBF model')
plt.show()

plt.scatter(x_test['Elev_Swath'], y_rbf, color='navy', lw=lw, label='RBF model')
plt.show()

plt.scatter(x_test['Elev_Swath'], y_test, color='navy', lw=lw, label='RBF model')
plt.show()
'''
