import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from datetime import datetime
import gc
import os

class RegressionScikit(object):
    def __init__(self):
        self._timeTaken = 0
        
    def train(self,x_train,y_train,model,dropCols=None,scaleY=True):
        
        startTime = time.time()
        self._dropCols = dropCols
        
        if self._dropCols  == None:
            self._x_train = x_train
        else:
            self._x_train = x_train.drop(dropCols,axis=1)
        self._y_train = y_train
        
        #Scale data
        self._xScaler = Scaler()
        self._xScaler.fit(self._x_train)
        self._x_train = self._xScaler.normalise(self._x_train)
        
        self._scaleY = scaleY
        if self._scaleY:
            self._yScaler = Scaler()
            self._yScaler.fit(self._y_train)
            self._y_train = self._yScaler.normalise(self._y_train)
        
        #Perform calc
        self._model = model
        self._model.fit(self._x_train,self._y_train)
        
        self._timeTaken = time.time()-startTime
        gc.collect()
        
    def predict(self,x_test):
        testSet = x_test
        if self._dropCols != None:
            testSet = x_test.drop(self._dropCols,axis=1)
        testSet = self._xScaler.normalise(testSet)
        yPred = self._model.predict(testSet)
        
        if self._scaleY:
            yPred = self._yScaler.denorm1D(yPred)
            
        return yPred
    
    @property
    def Y_train(self):
        return self._y_train
    
    @property
    def X_train(self):
        return self._x_train
    
    @property
    def TimeTaken(self):
        return self._timeTaken
    
    @property
    def Coeffs(self):
        return pd.DataFrame(zip(self._x_train.columns,self._model.coef_),columns=['Variable','Coeff'])
    
    @property
    def Intercept(self):
        return self._model.intercept_
    
    @property
    def Model(self):
        return self._model
    
    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self._model,path + '/model')
        self._saveParams(path+'/params.txt')
        dropCols = pd.DataFrame({'DropCols':self._dropCols})
        dropCols.to_hdf(path+'/dropCols.h5','DropCols')
        self._xScaler.save(path+'/xScaler')
        if self._scaleY:
            self._yScaler.save(path+'/yScaler')
        
    def _saveParams(self,path):
        text_file = open(path, "w")
        text_file.write("Model Params:\n")
        text_file.write("{}".format(self._model.get_params()))
        text_file.close()
        
    def load(self,path):
        self._model = joblib.load(path+'/model')
        self._dropCols = pd.read_hdf(path+'/dropCols.h5')['DropCols'].values.tolist()
        self._xScaler = Scaler()
        self._xScaler.load(path+'/xScaler')
        if os.path.isfile(path+'/yScaler'):
            self._scaleY = True
            self._yScaler = Scaler()
            self._yScaler.load(path+'/yScaler')
        else:
            self._scaleY = False
        
   
    
class RegressionTorch(object):
    def __init__(self):
        self._timeTaken = 0
    
    def train(self,x_train, y_train, dropCols=None,max_iter=500,lossFn='Huber',learningRate = 1e-2,optimizer="Adamax",scaleY=True):

        startTime = time.time()
        self._x_train = x_train
        self._y_train = y_train
               
        #If don't want to train on some columns then they need to be removed from data
        self._dropCols = dropCols
        if self._dropCols != None:
            self._x_train = self._x_train.drop(dropCols,axis=1)
        
        #Scale data
        self._xScaler = Scaler()
        self._xScaler.fit(self._x_train)
        self._x_train = self._xScaler.normalise(self._x_train)
        
        self._scaleY = scaleY
        if self._scaleY:
            self._yScaler = Scaler()
            self._yScaler.fit(self._y_train)
            self._y_train = self._yScaler.normalise(self._y_train)
               
        #Neural network model creation
        D_in = self._x_train.shape[1]
        D_out = 1 #Model only designed to predict 1 output at present
        H = D_in*D_in # Setting to the square may of the dimensions may be an overkill so can experiment with reducing if there are any performance issues
        
        self._model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.BatchNorm1d(H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H,H),
            #torch.nn.Linear(H,D_in),
            torch.nn.BatchNorm1d(H),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(H, D_out),
            #torch.nn.Linear(D_in, D_out),
        )
        
        
        #Check if GPU available
        self._applyCuda()

        #Save the pandas frames for later use
        self._x_trainPd = self._x_train
        self._y_trainPd = self._y_train

        #Convert to Torch datatypes - overwrite as don't want lots of data copies around
        self._x_train = torch.from_numpy(self._x_train.as_matrix())
        self._y_train = torch.from_numpy(self._y_train.as_matrix())
        self._x_train = Variable(self._x_train.type(self._dtype), requires_grad=False)
        self._y_train = Variable(self._y_train.type(self._dtype), requires_grad=False)
        
        #Set Loss Function
        if lossFn == 'Huber':
            self._loss_fn = torch.nn.SmoothL1Loss(size_average=True)
        elif lossFn == 'L1':
            self._loss_fn = torch.nn.L1Loss(size_average=False)
        elif lossFn == 'MSE':
            self._loss_fn = torch.nn.MSELoss(size_average=True)
        else: #Allow custom loss function to be passed in
            self._loss_fn = lossFn
        
        #Configure Optimizer
        learning_rate = learningRate
        if optimizer == 'Adamax':
            self._optimizer = torch.optim.Adamax(self._model.parameters(),lr=learning_rate,weight_decay=0,eps=1e-8,betas=(0.9,0.999))
        elif optimizer == 'Adam':
            self._optimizer = torch.optim.Adam(self._model.parameters(),lr=learning_rate,weight_decay=0,eps=1e-8,betas=(0.9,0.999))
        elif optimizer == 'SGD':
            self._optimizer = torch.optim.SGD(self._model.parameters(),lr=learning_rate)
        else: # Allow any optimizer combo to be passed in - needs good knowledge of model to do this
            self._optimizer = optimizer
        
        #Set limit to iterations
        maxIterations = max_iter
        
        #Perform calc
        self._model.train() # Only impacts batchnorm
        batchTime = time.time()
        for t in range(maxIterations):
            iB = 0
            batchSize = self._x_train.shape[0]//16
            while iB < self._x_train.shape[0]:
		xt = self._x_train[iB:iB+batchSize]
		#print(xt.shape[0])                
                # Forward pass: compute predicted y by passing x to the model.
                y_temp = self._model(xt)
                # Compute the loss
		yt = self._y_train[iB:iB+batchSize]
                loss = self._loss_fn(y_temp, yt)
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update
                self._optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                self._optimizer.step()
                iB+=batchSize
                
            if (t+1)%100 == 0:
                taken = time.time()-batchTime
                print("Progress: {:.0f}, BatchTime={:.4f}, Loss={}".format(t+1,taken,loss.data.cpu().numpy()[0]))
                batchTime = time.time()
        

        self._y_predTrain = y_temp
        
        self._timeTaken = time.time()-startTime
        gc.collect()
        
    def predict(self,x_test):
        self._model.eval() # Only impacts batchnorm
        testSet = x_test
        if self._dropCols != None:
            testSet = testSet.drop(self._dropCols,axis=1)
        testSet = self._xScaler.normalise(testSet)
        #Convert to numpy and then torch tensor type
        testSet = torch.from_numpy(testSet.as_matrix())
        testSet = Variable(testSet.type(self._dtype), requires_grad=False)
        #Calc prediction
        yPred = self._model(testSet)
        yPred = yPred.data.cpu().numpy().T[0]
        
        if self._scaleY:
            yPred = self._yScaler.denorm1D(yPred)
            
        return yPred

    @property
    def Y_predTrain(self):
        return self._y_predTrain.data.cpu().numpy().T[0] 
    
    @property
    def Y_train(self):
        return self._y_trainPd
          
    @property
    def X_train(self):
        return self._x_trainPd

    @property
    def Model(self):
        return self._model
    
    @property
    def Optim(self):
        return self._optimizer
    
    @property
    def TimeTaken(self):
        return self._timeTaken
    
    def _applyCuda(self):
        #Check if GPU available
        if torch.cuda.is_available():
            self._dtype = torch.cuda.FloatTensor
            self._model.cuda()
            print("Cuda available - feel the speed !!!!!!!!!!")
        else:
            self._dtype = torch.FloatTensor
            print("Cuda Unavailable - running on CPU which will be much slower")
        
    
    def save(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self._model,path + '/model')
        torch.save(self._optimizer.state_dict(),path + '/stateDict')
        self._saveParams(path+'/params.txt')
        dropCols = pd.DataFrame({'DropCols':self._dropCols})
        dropCols.to_hdf(path+'/dropCols.h5','DropCols')
        self._xScaler.save(path+'/xScaler')
        if self._scaleY:
            self._yScaler.save(path+'/yScaler')
        
    def _saveParams(self,path):
        text_file = open(path, "w")
        text_file.write("Optimiser Params:\n")
        text_file.write("{}".format(self._optimizer.state_dict()['param_groups']))
        text_file.write("\n\n------------------------------")
        text_file.write("\nHidden Structure:\n")
        text_file.write("{}".format(self._model.state_dict))
        text_file.close()
    
    def load(self,path):
        self._model = torch.load(path+'/model')
        self._applyCuda()
        self._dropCols = pd.read_hdf(path+'/dropCols.h5')['DropCols'].values.tolist()
        self._xScaler = Scaler()
        self._xScaler.load(path+'/xScaler')
        if os.path.isfile(path+'/yScaler'):
            self._scaleY = True
            self._yScaler = Scaler()
            self._yScaler.load(path+'/yScaler')
        else:
            self._scaleY = False

    
class Scaler(object):
    def __init__(self):
        self._scaler = preprocessing.StandardScaler()
        
    def fit(self,baseData):
        self._scaler.fit(self._r(baseData))
        
    def normalise(self,data):
        d = self._r(data)
        scaled = self._scaler.transform(d)
        newData = pd.DataFrame(scaled,columns=d.columns)
        return newData
    
    def denorm1D(self,data):
        d = self._r(data)
        unScaled = self._scaler.inverse_transform(d).ravel()
        return unScaled
    
    def _r(self,data):
        if data.ndim == 1:
            return pd.DataFrame(data) #Only scenario with 1 dim is for Elev_Oib - make more flexible in future
        return data
    
    def save(self,path):
        joblib.dump(self._scaler,path)
        
    def load(self,path):
        self._scaler = joblib.load(path)
    
    
