#!/bin/env python
import dhdt
import argparse
import netCDF4
import logging
from monitor import MonitorProcess
import time
from progress.bar import Bar

def mergeData(cfg,displayProgress=False):
    monitor = MonitorProcess()
    grid = dhdt.cfg2grid(cfg)
    tiles = dhdt.DHDTTiles(cfg)
    
    outnc = grid.netCDF(cfg['grid']['output'],store_tile_numbers = True,overwrite = True)
    outnc.registration = cfg['grid']['registration']

    if displayProgress:
        bar = Bar('merging tiles', max=tiles.nTiles)
    
    # loop over all tiles
    for i in range(tiles.nTiles):
        ta = time.time()
        fname = tiles.fname(i)
        idx = grid.bounds(i)
        outnc.variables['tile_number'][idx[1][0]:idx[1][1],idx[0][0]:idx[0][1]] = i
        tb = time.time()
        logging.debug("filling tile variable %f"%(tb-ta))
        try:
            nc = netCDF4.Dataset(fname,"r",format="NETCDF4",keepweakref=True)
        except:
            logging.warn('Cannot open %s'%fname)
            ta = time.time()
            logging.debug("opening tile (failed) %f"%(ta-tb))
            continue
        ta = time.time()
        logging.debug("opening tile %f"%(ta-tb))
        if nc.completed != 1:
            logging.warn('computations of tile %d are not completed'%i)
        if hasattr(nc,"registration") and nc.registration != cfg['grid']['registration']:
            logging.warn('mesh registration of tile %d does not match'%i)
        logging.debug("reading tile %s with %d time slices"%(fname,len(nc.dimensions['time'])))
        for k in range(len(nc.dimensions['time'])):
            outnc.variables['time'][k] =  nc.variables['time'][k]
            for v in outnc.variables.keys():
                if v not in ['x','y','time','lat','lon','pixelCompleted','tile_number']:
                    try: 
                        outnc.variables[v][k,idx[1][0]:idx[1][1],idx[0][0]:idx[0][1]] = nc.variables[v][k,:,:]
                    except:
                        logging.error('cannot read variable %s from tile %d'%(v,i))
        outnc.variables['pixelCompleted'][idx[1][0]:idx[1][1],idx[0][0]:idx[0][1]] = nc.variables['pixelCompleted'][:,:]
        outnc.sync()
        tb = time.time()
        logging.debug("copying data %f"%(tb-ta))
        nc.close()

        if displayProgress:
            bar.next()

    outnc.close()

    if displayProgress:
        bar.finish()
    
    peak = monitor.peak()
    logging.info('cpu: %.2f%%'%peak['cpu'])
    logging.info('rss: %.2fGB'%(peak['rss']/(1024.*1024.*1024.)))
    logging.info('vms: %.2fGB'%(peak['vms']/(1024.*1024.*1024.)))

def main():
    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-B','--display-progress-bar', default=False,action='store_true',help="display a progress bar")
    
    args = parser.parse_args()
    dhdt.initLog(args)
    
    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(args.config)

    if args.submit == 'sge':
        batch = dhdt.SGEProcess(args)
    elif args.submit == 'pbs':
        batch = dhdt.PBSProcess(args)
    else:
        batch = None
    
    if batch is not None:
        batch.serial(['mergeData',args.config])
    else:
        mergeData(cfg,displayProgress=args.display_progress_bar)

if __name__ == '__main__':
    main()
