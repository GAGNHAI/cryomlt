import multiprocessing
import numpy
import subprocess
import dhdt
import signal
import time
import sys, os.path
import logging
import tempfile
import string

class GracefulInterruptHandler(object):
    # from 
    # http://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True

        return True

class ProcessData(object):
    def __init__(self,cname,reprocess=True,monitor=False):
        self._cname = cname
        self._reprocess = reprocess
        self._monitor = monitor

    def __call__(self,n):
        args = ["processData","-p",str(n),self._cname]
        if self._reprocess:
            args.append('-r')
        if self._monitor:
             args.append('--monitor-memory')
        p = subprocess.Popen(args)
        with GracefulInterruptHandler() as h:
            while p.returncode is None:
                if h.interrupted:
                    p.terminate()
                p.poll()
                time.sleep(1)

def processMP(cname,np,runtime,reprocess=True,monitor=False):
    # extract run time
    rt = 0
    factor = 1
    tl = runtime.split(':')
    tl.reverse()
    if len(tl)>3:
        parser.error('cannot parse runtime argument %s'%(runtime))
    for ts in tl:
        try:
            t = int(ts)
        except:
            parser.error('cannot parse runtime argument %s'%(runtime))
        rt+=factor*t
        factor = factor*60
 

    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(cname)

    # create output directories
    dhdt.createOutDirs(cfg)

    # read the data
    if reprocess or not dhdt.checkStore(cfg):
        dhdt.readData(cfg)

    # process all data
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(processes=np)
    signal.signal(signal.SIGINT, original_sigint_handler)

    tiles = list(dhdt.DHDTTiles(cfg,reprocess=reprocess))

    logging.info("processing %d tiles"%len(tiles))

    if len(tiles)>0:

        f = ProcessData(cname,reprocess=reprocess,monitor=monitor)

        try:
            res = pool.map_async(f,tiles)
            res.get(rt)
        except KeyboardInterrupt:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    # merge data files
    dhdt.mergeData.mergeData(cfg)

def processSGE(cname,batch,reprocess=True,monitor=False,taskFarm=False):
    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(cname)

    # create the output directories
    cmd = ['createOutDirs',cname]
    jid = batch.serial(cmd)

    # read the data
    if cfg['data']['storeType'] == 'multihdf':
        # use multi-file backend
        cmd = ['readDataTF',cname]
        if reprocess:
            cmd.append('-r')
        llgrid = dhdt.cfg2llgrid(cfg)
        mname = os.path.join(cfg['data']['store'],'meta.json')
        if os.path.exists(mname) and not llgrid.check(mname):
            logging.error('multi-file store settings do not match')
            return
        jid = batch.mpi(cmd,njobs=min(batch.num_processes,llgrid.ncells),wait=jid)
    else:
        cmd = ['readData',cname]
        if reprocess or not dhdt.checkStore(cfg):
            jid = batch.serial(cmd,wait=jid)
        else:
            jid = None

    if taskFarm:
        # process data using task farm
        cmd = ['processDataTF',cname]
        if reprocess:
            cmd.append('-r')
        jid = batch.mpi(cmd,wait=jid)
    else:
        # process data using array jobs, one for each tile
        cmd=['processData',cname]
        if reprocess:
            cmd.append('-r')
        if monitor:
            cmd.append('--monitor-memory')
        cmd.append('-T')
        cmd.append('') # dummy tile filename
        tiles =  dhdt.DHDTTiles(cfg,reprocess=reprocess)
        jids=[]
        iterTiles = iter(tiles)
        allDone = False
        while not allDone:
            # open the tile file
            tFile = tempfile.NamedTemporaryFile(prefix=cfg['grid']['output']+"_tiles.",delete=False)
            nTiles = 0

            while nTiles < batch.max_array_size:
                try:
                    t = next(iterTiles)
                except:
                    allDone = True
                    break
                tFile.write("%d\n"%t)
                nTiles += 1

            cmd[-1] = tFile.name
            tFile.close()
            if nTiles>0:
                jids.append( batch.array(cmd,njobs=nTiles,wait=jid))

        jid = string.join(jids,',')
        if len(jid) == 0:
            jid = None

    # merge data files
    jid = batch.serial(['mergeData',cname],wait=jid)

def main():
    import argparse

    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--reprocess-data',action='store_true',default=False,help="process data even if previous run was successful")
    parser.add_argument('-n','--num-processes',type=int,default=8,help="set the number of processes to use (either on workstation or for MPI taskfarm)")
    parser.add_argument('--monitor-memory',action='store_true',default=False,help="monitor CPU and memory usage")
    parser.add_argument('-T','--task-farm',action='store_true',default=False,help="use MPI task farm")
    args = parser.parse_args()
    dhdt.initLog(args)

    if args.submit == 'sge':
        batch = dhdt.SGEProcess(args)
    elif args.submit == 'pbs':
        batch = dhdt.PBSProcess(args)
    else:
        batch = None
    
    if batch is not None:
        processSGE(args.config,batch,reprocess=args.reprocess_data,monitor=args.monitor_memory,taskFarm=args.task_farm)
    else:
        processMP(args.config,args.num_processes,args.run_time,reprocess=args.reprocess_data,monitor=args.monitor_memory)

if __name__ == '__main__':
    main()
