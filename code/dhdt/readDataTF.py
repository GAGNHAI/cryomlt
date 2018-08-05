#!/bin/env python
import dhdt
import logging
import sys
import numpy
import os.path
from readData import readData
from datastore import getStoreName
from tiles import Tiles

MIN_SIZE = 3

def readDataTF(cfg,rebuild=False):
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < MIN_SIZE:
        logging.error('need at least %d processes'%MIN_SIZE)
        return

    status = MPI.Status()

    if rank == 0:
        # MASTER
        lname = "MASTER"
        busy = numpy.ones(size,dtype=int)
        # the number of tasks
        nTasks = 0
        llgrid = dhdt.cfg2llgrid(cfg)
        if not llgrid.check(os.path.join(cfg['data']['store'],'meta.json')):
            logging.error('multifile DB settings do not match')
            nTasks = 0
        else:
            nTasks = llgrid.ncells

        tiles = Tiles(nTasks,os.path.join(getStoreName(cfg),'store.sqlite'),reprocess=rebuild)
        tilesIterator = iter(tiles)
            
        while numpy.any(busy>0):
            data = comm.recv(status=status)
            source = status.Get_source()
            if data == "get":
                try:
                    t= next(tilesIterator)
                except StopIteration:
                    t = None
                    busy[0] = 0
                    busy[source] = 0
                except:
                    logging.error(sys.exc_info()[0])
                    comm.Abort()
                comm.send(t,dest=source)
            else:
                tiles.updateTile(data)
                logging.info("%s:finished cell %d"%(lname,data))        
    else:
        # SLAVE
        lname = "SLAVE %d"%rank
        
        while True:
            comm.send("get",dest=0)
            cell = comm.recv(source=0)
            if cell is None:
                # done processing
                logging.info("%s:no more cells"%lname)
                break
            
            # do stuff
            logging.info("%s:computing cell %d"%(lname,cell))
            try:
                data, dataPoca = readData(cfg,cell=cell)
                # report done
                comm.send(cell,dest=0)
            except:
                logging.error(sys.exc_info()[0])
    
    comm.Barrier()
    logging.info(lname+":finished")    

def main():
    import argparse
    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser(taskfarm=True)])
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

    if batch is not None:
        cmd = ['readDataTF',args.config]
        if args.rebuild_store:
            cmd.append('-r')
        batch.mpi(cmd)
    else:
        readDataTF(cfg,rebuild=args.rebuild_store)


if __name__ == '__main__':
    main()
