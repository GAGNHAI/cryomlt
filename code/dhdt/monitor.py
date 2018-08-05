import psutil
import multiprocessing
import Queue
import time
import os

__all__ = ['MonitorProcess']

class MonitorWorker(multiprocessing.Process):
    def __init__(self,pid,tQ,rQ,out=None):
        self.tQ = tQ
        self.rQ = rQ
        self.p = psutil.Process(pid)
        
        self.pRSS = 0
        self.pVMS = 0
        self.pCPU = 0

        self.out = out

        multiprocessing.Process.__init__(self)
        self.daemon = True

    def run(self):
        if self.out is not None:
            fd = os.open( self.out, os.O_RDWR|os.O_CREAT, 0644 )
            os.write(fd,"#time CPU RSS VMS\n")
        ts = time.time()
        tstart = ts
        while True:
            m = None
            cpu = None
            if self.p.is_running():
                m = self.p.memory_info()
                cpu = self.p.cpu_percent()
                self.pRSS = max(self.pRSS,m.rss)
                self.pVMS = max(self.pVMS,m.vms)
                self.pCPU = max(self.pCPU,cpu)                

            t = time.time()
            if self.out is not None and t-ts>1:
                os.write(fd,"%.2f %.2f %d %d\n"%(t-tstart,cpu,m.rss,m.vms))
                ts = t

            task = None
            try:
                task = self.tQ.get(False)
            except Queue.Empty:
                pass
            
            if task == 'stop':
                if self.out is not None:
                    os.close(fd)
                return
            elif task == 'current':
                results = {}
                if m is not None and cpu is not None:
                    results['cpu'] = cpu
                    results['rss'] = m.rss
                    results['vms'] = m.vms
                else:
                    results['cpu'] = None
                    results['rss'] = None
                    results['vms'] = None
                self.rQ.put(results)
            elif task == 'peak':
                results = {}
                results['cpu'] = self.pCPU
                results['rss'] = self.pRSS
                results['vms'] = self.pVMS
                self.rQ.put(results)
            
            time.sleep(0.5)

class MonitorProcess(object):
    def __init__(self,pid=None,out=None):
        self.tQ = multiprocessing.Queue()
        self.rQ = multiprocessing.Queue()

        if pid is None:
            p = os.getpid()
        else:
            p = pid

        self.worker = MonitorWorker(p,self.tQ,self.rQ,out=out)
        self.worker.start()

    def __del__(self):
        self.tQ.put('stop')

    def current(self):
        self.tQ.put('current')
        return self.rQ.get()

    def peak(self):
        self.tQ.put('peak')
        return self.rQ.get()

if __name__ == '__main__':
    import sys
    if len(sys.argv)>1:
        pid = int(sys.argv[1])
    else:
        pid = None
    m = MonitorProcess(pid,"monitor.data")
    print m.current()
    for i in range(10):
        print m.current(),m.peak()
        time.sleep(1)
    print m.peak()
    
