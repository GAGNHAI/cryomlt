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

lw = 2
'''

lw = 2
#plt.scatter(X, y, color='darkorange', label='data')
#plt.scatter(X2, y, color='blue', label='data')
plt.scatter(y_test, rbfElev.Y_pred, color='navy', lw=lw, label='RBF Elev')
#plt.scatter(y_test, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('SVR')
plt.legend()
plt.show()
'''
'''



plt.scatter(y_train, nnResult.Y_predTrain, color='navy', lw=lw, label='Neural network Train')
#plt.scatter(y_test, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('Neural Network')
plt.legend()
plt.show()
'''


#plt.scatter(X, y, color='darkorange', label='data')
#plt.scatter(X2, y, color='blue', label='data')
plt.scatter(y_test, nnResult.Y_pred, color='navy', lw=lw, label='Neural network')
#plt.scatter(y_test, y_lin, color='c', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('Neural Network')
plt.legend()
plt.show()

plt.scatter(x_test['Coh_Swath'], y_rbf, color='navy', lw=lw, label='RBF model')
plt.show()

plt.scatter(x_test['Elev_Swath'], y_rbf, color='navy', lw=lw, label='RBF model')
plt.show()

plt.scatter(x_test['Elev_Swath'], y_test, color='navy', lw=lw, label='RBF model')
plt.show()

