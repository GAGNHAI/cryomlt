#!/bin/env python
import dhdt
from processData import processData
from tiles import DHDTTiles
import argparse
import numpy
import logging
import sys
from monitor import MonitorProcess

MIN_SIZE = 3

def processDataTF(cfg,monitor=False,reprocess=False):
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < MIN_SIZE:
        logging.error('need at least %d processes'%MIN_SIZE)
        return

    if monitor:
        mname = '%s_memmon.%06d'%(cfg['grid']['output'],rank)
    else:
        mname = None
    monitor = MonitorProcess(out=mname)
    
    status = MPI.Status()
    
    if rank == 0:
        # MASTER
        lname = "MASTER"
        logging.info("%s:started"%lname)
        busy = numpy.ones(size,dtype=int)

        tiles = DHDTTiles(cfg,reprocess=reprocess)
        tilesIterator = iter(tiles)
        
        while numpy.any(busy>0):
            data = comm.recv(status=status)
            source = status.Get_source()
            if data == "get":
                try:
                    t= next(tilesIterator)
                    logging.info("%s:shedule tile %d on %d"%(lname,t,source))
                except StopIteration:
                    t = None
                    busy[0] = 0
                    busy[source] = 0
                    logging.info("%s:shedule %d to shutdown"%(lname,source))
                except:
                    logging.error(sys.exc_info()[0])
                    comm.Abort()                
                comm.send(t,dest=source)
            else:
                tiles.updateTile(data)
                logging.info("%s:finished tile %d"%(lname,data))

    else:
        # SLAVE
        lname = "SLAVE %d"%rank
        logging.info("%s:started"%lname)
        while True:
            logging.info("%s:waiting for task"%lname)
            comm.send("get",dest=0)
            tile = comm.recv(source=0)
            if tile is None:
                # done processing
                logging.info("%s:no more tiles"%lname)
                break
            
            # do stuff
            logging.info("%s:computing tile %d"%(lname,tile))
            try:
                processData(cfg,tile,reprocess=reprocess)
                # report done
                comm.send(tile,dest=0)
            except:
                logging.error('processing tile %d %s'%(tile,sys.exc_info()[0]))
                
    comm.Barrier()

    peak = monitor.peak()
    logging.info('MPI: %d cpu: %.2f%%'%(rank,peak['cpu']))
    logging.info('MPI: %d rss: %.2fGB'%(rank,peak['rss']/(1024.*1024.*1024.)))
    logging.info('MPI: %d vms: %.2fGB'%(rank,peak['vms']/(1024.*1024.*1024.)))
    
    logging.info(lname+":finished")

def main():
    import argparse
    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser(taskfarm=True)])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--reprocess-data',action='store_true',default=False,help="process data even if previous run was successful")
    parser.add_argument('--monitor-memory',action='store_true',default=False,help="monitor CPU and memory usage")
    args = parser.parse_args()
    
    if args.submit == 'sge':
        batch = dhdt.SGEProcess(args)
    elif args.submit == 'pbs':
        batch = dhdt.PBSProcess(args)
    else:
        batch = None

    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(args.config)

    if batch is not None:
        dhdt.initLog(args)
        cmd = ['processDataTF',args.config,'-l',args.log_level,'-L',args.log_file]
        if args.reprocess_data:
            cmd.append('-r')
        if args.monitor_memory:
            cmd.append('--monitor-memory')
        batch.mpi(cmd)
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        dhdt.initLog(args,mpi_rank=rank)
        
        processDataTF(cfg,reprocess=args.reprocess_data,monitor=args.monitor_memory)
    
if __name__ == '__main__':
    main()
