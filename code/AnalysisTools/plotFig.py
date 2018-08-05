import matplotlib.pyplot as plt

fig = plt.figure()
plt.hist(x_test['Elev_Swath']-y_test,bins=50,range=[-500,500], figure=fig)

fig2 = plt.figure()
plt.hist(nnResult.Y_pred-y_test,bins=50,range=[-500,500], figure=fig2)