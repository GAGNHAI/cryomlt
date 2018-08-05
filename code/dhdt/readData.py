__all__ = ['readData']

from datastore import getStore, checkStore
from monitor import MonitorProcess
import logging

def readData(cfg,cell=-1):
    # read the data
    monitor = MonitorProcess()

    data, dataPoca = getStore(cfg,mode='w',cell=cell)
    if data is not None:
        data.readData(cfg['data']['input'])
    if dataPoca is not None:
        dataPoca.readData(cfg['data']['inputPoca'])

    peak = monitor.peak()
    logging.info('cpu: %.2f%%'%peak['cpu'])
    logging.info('rss: %.2fGB'%(peak['rss']/(1024.*1024.*1024.)))
    logging.info('vms: %.2fGB'%(peak['vms']/(1024.*1024.*1024.)))
    #logging.info('global bounding box %s',str(data.bbox))
    return data,dataPoca

def main():
    import dhdt
    import argparse

    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--rebuild-store',action='store_true',default=False,help="rebuild data store even though store is newer than the infput files")
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

    if not args.rebuild_store:
        if checkStore(cfg):
            logging.info('data store is up-to-date')
            return

    if batch is not None:
        if cfg['data']['storeType'] == 'dask':
            batch.omp(['readData',args.config])
        else:
            batch.serial(['readData',args.config])
    else:
        data,dataPoca = readData(cfg)


if __name__ == '__main__':
    main()
    
