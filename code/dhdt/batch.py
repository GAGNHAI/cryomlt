__all__ = ['MPI_SIZE','batchParser','BatchProcess']

import argparse

MPI_SIZE=8

BATCH_SYSTEMS = ['sge','pbs']

def batchParser(add_help=False,rt="4:00:00",max_array_size=10000,array=True,taskfarm=False):
    parser = argparse.ArgumentParser(add_help=add_help)
    group = parser.add_argument_group('batch processing options')
    group.add_argument('-s','--submit',metavar='BATCH',choices=BATCH_SYSTEMS,help="submit to a cluster using BATCH system, can be one of %s"%str(BATCH_SYSTEMS))
    group.add_argument('-t','--run-time',metavar="hh:mm:ss",default=rt,help="set the maximum runtime for processing each tile. The minutes or hours do not need to be specified.  default=%s"%rt)
    group.add_argument('-m','--memory',metavar="MEM",default="4G",help="amount of memory per job, default 4G")
    group.add_argument('-w','--wait',metavar='JID',help="wait for job JID before executing command")
    if array:
        group.add_argument('--max-array-size',metavar="N",type=int,default=max_array_size,help="limit the array job size to N. Submit multiple array jobs if N<number of tiles. Default N=%d"%max_array_size)
    if taskfarm:
        group.add_argument('-n','--num-processes',type=int,default=MPI_SIZE,help="set the number of processes to use (default %d)"%MPI_SIZE)
    return parser

class BatchProcess(object):
    def __init__(self,args):
        for a in ['run_time','memory','wait','max_array_size','num_processes']:
            if hasattr(args,a):
                setattr(self,'_'+a,getattr(args,a))
            else:
                setattr(self,'_'+a,None)

    @property
    def run_time(self):
        return self._run_time
    @property
    def memory(self):
        return self._memory
    @property
    def wait(self):
        return self._wait
    @property
    def max_array_size(self):
        return self._max_array_size
    @property
    def num_processes(self):
        return self._num_processes

    def serial(self,cmd,wait=None):
        raise NotImplementedError

    def array(self,cmd,njobs,wait=None):
        raise NotImplementedError

    def mpi(self,cmd,njobs=None,wait=None):
        raise NotImplementedError

    def omp(self,cmd,wait=None):
        raise NotImplementedError

if __name__ == '__main__':

    parser=batchParser(add_help=True)
    args = parser.parse_args()

    print args.submit
