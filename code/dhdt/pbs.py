__all__ = ['PBSProcess']

import subprocess
import os.path,sys
from batch import *

class PBSProcess(BatchProcess):
    def wait_option(self,wait=None):
        if wait is not None:
            if self.wait is not None:
                w = self.wait + ':' + wait
            else:
                w = wait
        else:
            w = self.wait
        if w is not None:
            return ['-W','depend=afterok:'+w]
        else:
            return []
            
    def serial(self,cmd,wait=None):
        c = ['qsub','-A','d43-gourmelen','-l','select=serial=true:ncpus=1','-l','walltime=%s'%self.run_time,'-N',cmd[0]] + self.wait_option(wait=wait)
        p = subprocess.Popen(c,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        p.stdin.write("""
#!/bin/bash

module load anaconda-compute
source activate dhdtPy-dev

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)               

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

%s
"""%" ".join(cmd))

        stdout,stderr = p.communicate()
        if p.returncode == 0:
            jobid = stdout.strip()
            return jobid
        else:
            raise RuntimeError, stderr

    def array(self,cmd,njobs,wait=None):
        assert njobs <= self.max_array_size
        c = ['qsub','-r','y','-J','1-%d'%njobs,'-A','d43-gourmelen','-l','select=1','-l','walltime=%s'%self.run_time,'-N',cmd[0]] + self.wait_option(wait=wait)
        p = subprocess.Popen(c,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        p.stdin.write("""
#!/bin/bash

module load anaconda-compute
module swap PrgEnv-cray PrgEnv-gnu
module load cray-mpich
source activate dhdtPy-dev

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)               

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

let p=PBS_ARRAY_INDEX-1

%s -p $p
"""%" ".join(cmd))

        stdout,stderr = p.communicate()
        if p.returncode == 0:
            jobid = stdout.strip()
            return jobid
        else:
            raise RuntimeError, stderr

    def mpi(self,cmd,njobs=None,wait=None):
        if njobs is None:
            njobs = self.num_processes
        assert njobs is not None
        nnodes = njobs//24
        if njobs%24 > 0:
            nnodes += 1
        c = ['qsub','-r','y','-A','d43-gourmelen','-l','select=%d'%nnodes,'-l','walltime=%s'%self.run_time,'-N',cmd[0]] + self.wait_option(wait=wait)
        p = subprocess.Popen(c,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        p.stdin.write("""
#!/bin/bash

module load anaconda-compute
source activate dhdtPy-dev

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)               

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

aprun -n %d %s 
"""%(njobs," ".join(cmd)))
        
        stdout,stderr = p.communicate()
        if p.returncode == 0:
            jobid = stdout.strip()
            return jobid
        else:
            raise RuntimeError, stderr

    def omp(self,cmd,wait=None):
        c = ['qsub','-A','d43-gourmelen','-l','select=1','-l','walltime=%s'%self.run_time,'-N',cmd[0]] + self.wait_option(wait=wait)
        p = subprocess.Popen(c,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        p.stdin.write("""
#!/bin/bash

module load anaconda-compute
source activate dhdtPy-dev

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)               

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

aprun -n 1 -d 24 %s
"""%" ".join(cmd))

        stdout,stderr = p.communicate()
        if p.returncode == 0:
            jobid = stdout.strip()
            return jobid
        else:
            raise RuntimeError, stderr
        
        
if __name__ == '__main__':

    parser=batchParser(add_help=True)
    args = parser.parse_args()

    print args
    
