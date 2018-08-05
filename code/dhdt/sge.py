__all__ = ['SGEProcess']

import subprocess
import os.path,sys
from batch import *

class SGEProcess(BatchProcess):
    def __init__(self,args,pe='scatter'):
        BatchProcess.__init__(self,args)

        self._pe = pe
        
        # find SGE scripts
        self._qscripts = {}
        for s in ['qsub_dhdt.sh','qsub_dhdt_mpi.sh']:
            dir=os.path.join(os.path.dirname(sys.argv[0]),'..')
            for d in [os.path.join(dir,'share','dhdtPy'),os.path.join(dir,'qsub')]:
                qscript = os.path.abspath(os.path.join(d,s))
                if os.path.isfile(qscript):
                    self._qscripts[s] = qscript
                    break
            else:
                raise RuntimeError,'could not find qsub script %s'%s
    @property
    def pe(self):
        return self._pe
            
    def wait_option(self,wait=None):
        if wait is not None:
            if self.wait is not None:
                w = self.wait + ',' + wait
            else:
                w = wait
        else:
            w = self.wait
        if w is not None:
            return ['-hold_jid',w]
        else:
            return []
            
    def serial(self,cmd,wait=None):
        c = ['qsub','-pe',self.pe,'1','-l','h_rt=%s'%self.run_time,'-l','h_vmem=%s'%self.memory,'-N',cmd[0]] + self.wait_option(wait=wait)
        c.append(self._qscripts['qsub_dhdt.sh'])
        c += cmd
        p = subprocess.check_output(c)
        return p.split()[2].split('.')[0]

    def array(self,cmd,njobs,wait=None):
        assert njobs <= self.max_array_size
        c = ['qsub','-pe',self.pe,'1','-l','h_rt=%s'%self.run_time,'-l','h_vmem=%s'%self.memory,'-N',cmd[0]] + self.wait_option(wait=wait)
        c += ['-t','1-%d'%njobs]
        c.append(self._qscripts['qsub_dhdt.sh'])
        c += cmd
        p = subprocess.check_output(c)
        return p.split()[2].split('.')[0]

    def mpi(self,cmd,njobs=None,wait=None):
        if njobs is None:
            njobs = self.num_processes
        assert njobs is not None

        c = ['qsub','-pe',self.pe,str(njobs),'-l','h_rt=%s'%self.run_time,'-l','h_vmem=%s'%self.memory,'-N',cmd[0]] + self.wait_option(wait=wait)
        c.append(self._qscripts['qsub_dhdt_mpi.sh'])
        c += cmd
        p = subprocess.check_output(c)
        return p.split()[2].split('.')[0]
    
if __name__ == '__main__':

    parser=batchParser(add_help=True)
    args = parser.parse_args()

    print args
    
