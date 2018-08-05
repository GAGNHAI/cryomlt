#__all__ = ['surfaceFit']

import numpy,numpy.ma,numpy.linalg
import scipy.sparse,scipy.sparse.linalg
import time
import logging
import pandas as pd

class Model:
    """ Class defining the model options:
        - type
        - lew_where
        - lew_opt
        - lew_prop
        - lew_timeCorr
        - pow_where
        - pow_opt
        - pow_timeCorr """
        
        
    def __init__(self):
        self.type = 'linear'
        self.weight = 'power4'
        self.lew_where = 0
        self.lew_opt = None
        self.lew_prop = 0.
        self.lew_timeCorr = 0
        self.pow_where = 0
        self.pow_opt = None
        self.pow_timeCorr = 0

def computeWeight(d, weight='power4'):
    
    
    if weight == 'power4':
        power = d.power.values*d.power.values
        w = power/max(power)
        w = w*w
    elif weight == 'ones':
        w = numpy.ones(d.power.size,dtype=float)
    else:
        raise RuntimeError, 'unknown weight function: %s' %weight
    
    
    return w
    
def averageAccordingToTimeDelay(vec1,t,w,opt = None, timeCorr = 0):
     

    if (opt is not None) and (opt != 'none'): 
        if opt == 'yes':
            lew = vec1
        elif opt == 'mean':
            lew = numpy.zeros(vec1.size,dtype=float)
            for ind in range(vec1.size):
                timeRef = t[ind]
                index = numpy.where( (numpy.abs(t-timeRef)<=timeCorr) & (w>0) )
                lew[ind] = numpy.mean(vec1[index[0]])
        else:
            print opt
            raise ValueError, 'wrong value for opt'
    else:
        lew = numpy.zeros(w.size,dtype=float)
        
  
    return lew

def surfaceFit(d,model = Model()):
    
    # get initial weights
    w = computeWeight(d, model.weight)
    
    # mask huge elevation variation within one pixel
    z = d.elev.values
    mask = abs(z-numpy.median(z))>3*numpy.std(z)
    modified_weights = numpy.where(mask,0,w)

    # the number of measurements used
    nMeasurements = numpy.count_nonzero(abs(modified_weights))
    
    nIter = 0
    tstart = time.time()
    while nMeasurements > 0:
        # fit data
        z_fit,coeff, cov_matrix = fit_model(d,modified_weights,model=model)
    
        # standard deviation of residuals
        # mask out the values that are not used
        residuals = numpy.ma.array(z-z_fit,mask=mask)
        std = numpy.std(residuals)
        # reject measurements whose residual is bigger than 3 times the std of 
        # the residuals (3-sigma editing, as in Flament and Remy, 2012
        tmpMask = abs(residuals)>3*std
        if not numpy.any(tmpMask):
            # fit converged
            break

        # update the mask, the weights and the number of measurements
        mask = numpy.where(tmpMask,True,mask)
        modified_weights = numpy.where(mask,0,w)
        nMeasurements = numpy.count_nonzero(abs(modified_weights))
        nIter = nIter+1

    # produce masked arrays
    z_fit = numpy.ma.array(z_fit,mask=mask)
    maskedX = numpy.ma.array(d.x.values,mask=mask)
    maskedY = numpy.ma.array(d.y.values,mask=mask)
    maskedZ = numpy.ma.array(d.elev.values,mask=mask)
    maskedT = numpy.ma.array(d.time.values,mask=mask)

    std_res = numpy.std(maskedZ-z_fit)

    logging.debug("fitting %d data using %d iteration took %f seconds",nMeasurements,nIter,time.time()-tstart)

    return (maskedX,maskedY,maskedZ,maskedT,mask,
            std_res,
            z_fit,coeff, cov_matrix)

def fit_model(d,w,model=Model()):

    z = d.elev.values
    t = d.time.values
    x = d.x.values
    y = d.y.values
    c = numpy.ones(x.size,dtype=float)
    
    
    if (model.lew_opt == 'model'):
        lew = d.leadEdgeW.values
    else:
        lew = numpy.zeros(x.size,dtype=float)
    
    if (model.pow_opt == 'poca') :  
        power = numpy.zeros(x.size,dtype=float)
        powerPoca = d.poca_power.values
    elif (model.pow_opt == 'power') :    
        power = d.power.values    
        powerPoca = numpy.zeros(x.size,dtype=float)
    elif (model.pow_opt == 'powerAndPoca'):
        power = d.power.values 
        powerPoca = d.poca_power.values
    else:
        power = numpy.zeros(x.size,dtype=float)
        powerPoca = numpy.zeros(x.size,dtype=float)
        
    if model.type =='quadratic':
        x2 = x*x
        y2 = y*y
        xy = x*y
    else:
        x2 = numpy.zeros(x.size,dtype=float)
        y2 = numpy.zeros(x.size,dtype=float)
        xy = numpy.zeros(x.size,dtype=float)
        
    matModel = numpy.mat([t,lew,x,y,x2,y2,xy,c,power,powerPoca]).T

    ws = scipy.sparse.diags(w,0,dtype=float)

    A=matModel
    b=numpy.mat(z).T

    coeff, cov_matrix = model_resolution(A,b,ws)
    
    if (model.lew_opt != 'model'):
        coeff[1] = -1e6 # filling value
        
    if (model.pow_opt != 'power') and (model.pow_opt != 'powerAndPoca'):
        coeff[8] = -1e6 # filling value

    if (model.pow_opt != 'poca') and (model.pow_opt != 'powerAndPoca'):
        coeff[9] = -1e6 # filling value

    if model.type == 'linear' :
        coeff[4] = -1e6 # filling value
        coeff[5] = -1e6 # filling value
        coeff[6] = -1e6 # filling value
          
    z_fit = numpy.squeeze(numpy.asarray(matModel*coeff))

    return z_fit,coeff, cov_matrix

def model_resolution(G,d,w):
    """compute resolution matrix"""

    # compute the inverse of the model matrix
    wsqrt = numpy.sqrt(w)
    G_inv = numpy.linalg.pinv(wsqrt*G)*wsqrt

    # compute the model coefficients
    coeff = G_inv*d

    d_modelled = G*coeff
    d_diff = d-d_modelled
    d_diff = d_diff*d_diff.T
    varData = numpy.diag(numpy.diag(d_diff))
    # calculate unit model covariance matrix
    cov_matrix = G_inv*varData*G_inv.T
    
    return (coeff, cov_matrix)

if __name__ == '__main__':
    import numpy.random

    stdn = 10;
    at=1.7
    ax=2.3
    ay=-2.
    alew = 0
    ac = 1
    nzero = 10


    x=100*numpy.random.randn(nzero)
    y=100*numpy.random.randn(nzero)
    t=10*numpy.random.randn(nzero)
    w = 4*numpy.ones(nzero,dtype=float)
    lew = numpy.absolute(10*numpy.random.randn(nzero))
    p = numpy.absolute(10*numpy.random.randn(nzero))
    n = stdn*numpy.random.randn(nzero)

    z=at*t+alew*lew+ax*x+ay*y+ac+n
    
    d = {'time':t,'x':x,'y':y,'leadEdgeW':lew,'elev':z,'power':p}
    df = pd.DataFrame(d)

    test = Model()
    test.lew_opt = 'mean'
    test.lew_timeCorr = 1
    z_fit,coeff, cov_matrix = fit_model(df,w,model=test)
    (maskedX,maskedY,maskedZ,maskedT,mask,std_res,z_fit,coeff, cov_matrix) = surfaceFit(df,model = test)

    print test.type
    print test.lew_where
    print coeff[1]
    

    vec1 = numpy.random.randn(nzero)
    t = numpy.arange(nzero)
    w = numpy.ones(vec1.size)
    vecOut = averageAccordingToTimeDelay(vec1,t,w,model=test,timeCorr=3600)
    print vec1
    print vecOut