#!/bin/env python
import dhdt
import argparse

import pandas as pd
from shapely.geometry import Point
import numpy
import datetime
import logging
import time
import sys
from monitor import MonitorProcess

def incrementTime(t,tstep,unit,timeOrigin):
    tn = None
    tt = timeOrigin+datetime.timedelta(seconds=t)
    if unit=='d':
        tn = tt+datetime.timedelta(tstep)
    elif unit=='m':
        tn =  tt.replace(year=tt.year+(tt.month-1+tstep)//12,month=(tt.month-1+tstep)%12+1)
    elif unit=='a':
        tn =  tt.replace(year=tt.year+tstep)
    else:
        raise RuntimeError, 'unknown time unit: %s'%unit
    return (tn-timeOrigin).total_seconds()
    
def computeTopo(pnt,t,d,model,coeff):
    
    index = numpy.argmin(numpy.abs(t-d.time.values))
    if (model.lew_opt != 'none') and (model.lew_opt is not None):
        value = d.leadEdgeW.values[index]
        if model.lew_opt == 'fix':
            lew_fix = value
            lew = 0.
        else:
            lew_fix = 0.
            lew = value
    else:
        lew = 0.
        lew_fix = 0.
        
    if (model.pow_opt == 'power') | (model.pow_opt == 'powerAndPoca'):
        power = d.power.values[index]
    else:
        power = 0.
    
    if (model.pow_opt == 'poca') | (model.pow_opt == 'powerAndPoca'):
        power_poca = d.poca_power.values[index]
    else:
        power_poca = 0.

    if model.type == 'linear' :
        gtopo = numpy.mat([t,lew,pnt.x,pnt.y,0.,0.,0.,1.,power,power_poca])
    elif model.type=='quadratic':
        gtopo = numpy.mat([t,lew,pnt.x,pnt.y,pnt.x*pnt.x, pnt.y*pnt.y,pnt.y*pnt.x,1.,power,power_poca])
    else:
        raise ValueError, 'unknown model type %s'%model
    
    topo = numpy.asscalar(gtopo*coeff)   
    topo = topo - model.lew_prop*lew_fix
        
    return topo

def processPoint(gdf,pnt,t,timeResolution,startTime,model=dhdt.Model(),minData=10, yearLength=365.2422,maxRate=1e5):
    # construct time interval
    d0 = incrementTime(t,-timeResolution['value'],timeResolution['units'],startTime)
    d1 = incrementTime(t,timeResolution['value'],timeResolution['units'],startTime)

    # select data that fall into circle and time interval
    tstart = time.time()
    d= gdf[(gdf['time'] >=  d0) & 
           (gdf['time'] <=  d1)]
    logging.debug("selecting data took %f seconds",time.time()-tstart)

    if d.shape[0]<minData:
        logging.debug('not enough data selected')
        return 

    if  model.lew_opt == 'fix':
        d['elev'] = d.elev.values + model.lew_prop*d.leadEdgeW.values

    (x,y,z,times,mask,std_res,
     z_fit,coeff,cov_matrix) = dhdt.surfaceFit(d,model=model)

    nMeasurements = mask.size
    nMeasurementsUsed = nMeasurements-numpy.count_nonzero(mask)

    if nMeasurementsUsed < minData:
        logging.debug('not enough data used')
        return

    # scale to m/a 
    rate = coeff[0,0]*yearLength*24*3600
    error =  numpy.sqrt(numpy.diag(cov_matrix)[0])*yearLength*24*3600
    
    # leading edge width
    coeffOut= numpy.squeeze(numpy.asarray(coeff[1::,0]))

    if abs(rate) > maxRate:
        logging.debug('abs(rate) %f>%f',abs(rate),maxRate)
        return
 

    topo = computeTopo(pnt,t,d,model,coeff)
    
    times = numpy.ma.array(d.time.values,mask=mask)
    

    return (topo,rate,std_res,times.min(),times.max(),
            nMeasurements,nMeasurementsUsed,
            error,coeffOut)

def processData(cfg,process,monitor=False,reprocess=False):
    logging.info("processing tile %d"%process)
    if monitor:
        mname = '%s_memmon.%06d'%(cfg['grid']['output'],process)
    else:
        mname = None
    monitor = MonitorProcess(out=mname)

    if process <0 or process>=cfg['misc']['nproc'][0]*cfg['misc']['nproc'][1]:
        raise ValueError,'wrong process number 0<=N<%d'%(cfg['misc']['nproc'][0]*cfg['misc']['nproc'][1])

    grid = dhdt.cfg2grid(cfg)
    
    #initialize model option     
    model = dhdt.Model() 
    model.type = cfg['dhdt']['model']   
    model.weight =  cfg['dhdt']['weight']   
    if cfg['data']['inputPoca'] != 'none':
        model.lew_where =  cfg['dhdt']['lew_where']
        model.lew_opt =  cfg['dhdt']['lew_opt']   
        model.lew_timeCorr =  cfg['dhdt']['lew_timeCorr'] 
        model.lew_prop =  cfg['dhdt']['lew_prop']
        model.pow_where =  cfg['dhdt']['pow_where']
        model.pow_opt =  cfg['dhdt']['pow_opt']   
        model.pow_timeCorr =  cfg['dhdt']['pow_timeCorr'] 
        
    
    #select data in tile
    radius = numpy.maximum(cfg['dhdt']['lew_where'],cfg['dhdt']['pow_where'])
    margin = cfg['dhdt']['radius']+radius

    nameListPoca = []
    nameListSwath = []

    dataSwath,dataPoca = dhdt.getStore(cfg,mode='r')

    pSwath = None
    pPoca  = None
    if dataSwath is not None:
        gSwath = dataSwath.getGeoPandas(crs=cfg['data']['projection'],bbox=grid.bboxGeo(process,margin=cfg['dhdt']['radius']),
                              minPower=cfg['dhdt']['minPower'],spass=cfg['dhdt']['satellitePass'],
                              demDiffMadThresh=cfg['dhdt']['maxDemDiffMad'],demDiffThresh=cfg['dhdt']['maxDemDiff'])
        if gSwath is None:
            pSwath = None
        else:
            tstart=time.time()
            pSwath = gSwath.to_crs(cfg['grid']['projection'])
            logging.debug('projecting data took %f seconds',time.time()-tstart)

            # make the coordinates easily accessible 
            tstart = time.time()
            pSwath['x'] = pSwath.centroid.map(lambda pnt: pnt.x)
            pSwath['y'] = pSwath.centroid.map(lambda pnt: pnt.y)
            logging.debug('mapping coordinates took %f seconds',time.time()-tstart)
            pSwath= pSwath.reset_index(drop=True)
                              
    if dataPoca is not None:
        gPoca = dataPoca.getGeoPandas(crs=cfg['data']['projection'],bbox=grid.bboxGeo(process,margin=margin),
                              minPower=cfg['dhdt']['minPower'],spass=cfg['dhdt']['satellitePass'],
                              demDiffMadThresh=cfg['dhdt']['maxDemDiffMad'],demDiffThresh=cfg['dhdt']['maxDemDiff']) 
        if gPoca is None:
            pPoca = None
        else:
            tstart=time.time()
            pPoca = gPoca.to_crs(cfg['grid']['projection'])
            logging.debug('projecting Poca data took %f seconds',time.time()-tstart)

            # make the coordinates easily accessible 
            tstart = time.time()
            pPoca['x'] = pPoca.centroid.map(lambda pnt: pnt.x)
            pPoca['y'] = pPoca.centroid.map(lambda pnt: pnt.y)
            logging.debug('mapping pPoca coordinates took %f seconds',time.time()-tstart)
            pPoca = pPoca.reset_index(drop=True)
            pPoca['poca_power'] = pPoca['power']

            if (model.lew_where > 0) | (model.lew_timeCorr>0):
                pPoca = dhdt.averageSpaceTime(pPoca, 'leadEdgeW', spaceCorr=model.lew_where, timeCorr = model.lew_timeCorr)

            if ((model.pow_opt == 'poca') | (model.pow_opt == 'powerAndPoca')) & ((model.pow_where > 0) | (model.pow_timeCorr>0)):
                pPoca = dhdt.averageSpaceTime(pPoca, 'poca_power', spaceCorr=model.pow_where, timeCorr = model.pow_timeCorr )   

    
    # create netCDF output file
    nc = grid.netCDF(cfg['grid']['output'],process,registration=cfg['grid']['registration'],overwrite=reprocess)
    
    if pSwath is None and pPoca is None:
        # no data
        nc.variables['pixelCompleted'][:,:] = 2
    else:    
                
        if dataSwath is not None and dataPoca is not None:
            nameListPoca = ['leadEdgeW','poca_power']
            nameListSwath = ['leadEdgeW','poca_power']
            pSwath = dhdt.linkPocaSwath(pPoca,pSwath,nameListPoca,nameListSwath)
            pSwath= pSwath.dropna(axis = 0, how = 'any',subset = nameListSwath )

        # choose data on which the dhdt is done
        if dataSwath is not None:
            p = pSwath
            data = dataSwath
        elif dataPoca is not None:
            p = pPoca
            data = dataPoca
        else:
            raise Exception('data should be provided')

        logging.debug('total number of measurements %d'%p.shape[0])
    
        # get the start and end date
        startDate = data.time(cfg['time']['period'][0])
        endDate = p.time.max()
        if cfg['time']['period'][1] is not None:
            endDate = min(data.time(cfg['time']['period'][1]),endDate)
        timeResolution = cfg['time']['timeResolution']

        k=0
        t = startDate
        while t<endDate:
            nc.variables['time'][k] = t
            if cfg['time']['timeStep'] is None:
                break
            t = incrementTime(t,cfg['time']['timeStep']['value'],cfg['time']['timeStep']['units'],data.timeOrigin)
            k+=1

        if cfg['grid']['registration'] == 'pixel':
            dx = 0.5 * cfg['grid']['posting']
            dy = 0.5 * cfg['grid']['posting']
        else:
            dx = 0.
            dy = 0.

        for j in range(len(nc.dimensions['y'])):
            for i in range(len(nc.dimensions['x'])):
                if nc.variables['pixelCompleted'][j,i] > 0:
                    logging.debug("already processed pixel (%d,%d)"%(j,i))
                    continue
                k=0
                t = startDate
                pnt = Point( nc.variables['x'][i]+dx,nc.variables['y'][j]+dy)
                circle =  pnt.buffer(cfg['dhdt']['radius'])
                tstart=time.time()

                selectedData = p[p['geometry'].within(circle)]
                logging.debug("preselection took %f seconds",time.time()-tstart)

                if selectedData.shape[0] < cfg['dhdt']['minData']:
                    logging.debug('no points selected')
                    nc.variables['pixelCompleted'][j,i] = 2
                    continue

                while t<endDate:
                    res = processPoint(selectedData,pnt,t,
                                       timeResolution,data.timeOrigin,
                                       model=model,
                                       minData=cfg['dhdt']['minData'],
                                       yearLength=cfg['time']['yearLength'],
                                       maxRate=cfg['dhdt']['maxRate'])

                    if res is not None:
                        nc.variables['topo'][k,j,i] = res[0]
                        nc.variables['rate'][k,j,i] = res[1]
                        nc.variables['stdResidual'][k,j,i] = res[2]
                        nc.variables['start'][k,j,i] = res[3]
                        nc.variables['end'][k,j,i] = res[4]
                        nc.variables['nMeasurements'][k,j,i] = res[5]
                        nc.variables['nMeasurementsUsed'][k,j,i] = res[6]
                        nc.variables['errorCovMat'][k,j,i] = res[7]
                        nc.variables['coeffLeadingEdgeWidth'][k,j,i] = res[8][0]
                        nc.variables['coeffX'][k,j,i] = res[8][1]
                        nc.variables['coeffY'][k,j,i] = res[8][2]
                        nc.variables['coeffX2'][k,j,i] = res[8][3]
                        nc.variables['coeffY2'][k,j,i] = res[8][4]
                        nc.variables['coeffXY'][k,j,i] = res[8][5]
                        nc.variables['coeffC'][k,j,i] = res[8][6]
                        nc.variables['coeffPower'][k,j,i] = res[8][7]  
                        nc.variables['coeffPowerPoca'][k,j,i] = res[8][8]  

                    if cfg['time']['timeStep'] is None:
                        break
                    t = incrementTime(t,cfg['time']['timeStep']['value'],cfg['time']['timeStep']['units'],data.timeOrigin)

                    k+=1
                nc.variables['pixelCompleted'][j,i] = 3
            nc.sync()
            
    nc.completed = 1
    nc.close()

    peak = monitor.peak()
    logging.info('cpu: %.2f%%'%peak['cpu'])
    logging.info('rss: %.2fGB'%(peak['rss']/(1024.*1024.*1024.)))
    logging.info('vms: %.2fGB'%(peak['vms']/(1024.*1024.*1024.)))

def main():

    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--reprocess-data',action='store_true',default=False,help="process data even if previous run was successful")
    parser.add_argument('--process','-p',metavar='N',default=0,type=int,help="compute tasks for process N")
    parser.add_argument('--tile-file','-T',metavar='TFILE',help="get tile IDs from file TFILE")
    parser.add_argument('--monitor-memory',action='store_true',default=False,help="monitor CPU and memory usage")
    args = parser.parse_args()
    dhdt.initLog(args)

    if args.submit == 'sge':
        batch = dhdt.SGEProcess(args)
    elif args.submit == 'pbs':
        batch = dhdt.PBSProcess(args)
    else:
        batch = None
    
    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(args.config)

    processTile = args.process
    if args.tile_file is not None:
        i = 0
        tfile = open(args.tile_file,'r')
        for line in tfile.readlines():
            if i == args.process:
                processTile = int(line)
                break
            i = i+1
        else:
            parser.error("could not find tile %d in tile file %s"%(args.process,args.tile_file))
        tfile.close()

    if not args.reprocess_data:
        if dhdt.checkNC(cfg['grid']['output'],processTile):
            logging.info('tile %d has already been successfully processed'%processTile)
            return

    if batch is not None:
        cmd = ['processData','-p',str(processTile),args.config,'-l',args.log_level,'-L',args.log_file]
        if args.monitor_memory:
            cmd.append('--monitor-memory')
        if args.reprocess_data:
            cmd.append('-r')
        batch.serial(cmd)
    else:
        processData(cfg,processTile,monitor=args.monitor_memory,reprocess=args.reprocess_data)


if __name__ == '__main__':
    main()
