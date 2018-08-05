__all__ = ['Tiles','DHDTTiles']

from grid import ncName
import logging
import netCDF4
import sqlite3
import numpy
import time

class Tiles(object):
    def __init__(self,nTiles,dbName,reprocess=False):
        self._reprocess = reprocess
        self._nTiles = nTiles
        self._ts = time.time()
        
        self._conn = sqlite3.connect(dbName)
        self._c = self._conn.cursor()
        if reprocess:
            self._c.execute("drop table if exists tiles")
        self._c.execute("create table if not exists tiles (tile integer primary key, done real, ts real)")
        self._conn.commit()

    def __del__(self):
        self._conn.close()

    @property
    def reprocess(self):
        return self._reprocess
        
    @property
    def nTiles(self):
        return self._nTiles
    @property
    def ts(self):
        return self._ts

    def updateTile(self,i,done=1.):
        self._c.execute("insert or replace into tiles values (?,?,?)",(i,done,time.time()))
        self._conn.commit()

    def statusTile(self,i):
        """check status of a tile

        return 0 if not started
               1 if partial
               2 if done
        """
        if i<0 or i> self.nTiles-1:
            raise IndexError, "index out of range"
        self._c.execute("select done,ts from tiles where tile=?",(i,))
        res = self._c.fetchone()
        if res is None:
            # not started yet
            return 0
        else:
            if abs(res[0]-1) < 1e-10:
                # done
                return 2
            else:
                return 1

    def completed(self,i):
        s = self.statusTile(i)
        if s==0:
            return 0.
        elif s==2:
            return 1.
        else:
            self._c.execute("select done from tiles where tile=?",(i,))
            res = self._c.fetchone()
            return res[0]
            
    def totalCompleted(self):
        self._c.execute("select sum(done) from tiles")
        total = self._c.fetchone()[0]
        if total is None:
            return 0.
        else:
            return total/float(self.nTiles)
            
    def __len__(self):
        return sum(1 for _ in self)

    def noUpdateIter(self):
        c = self._conn.cursor()
        c.execute("select tile from tiles where done<1")
        for res in c:
            yield res[0]
    
    def __iter__(self):
        for t in range(self.nTiles):
            s = self.statusTile(t)
            if s == 2:
                logging.debug("tile %d is already complete"%t)
                continue
            elif s == 1:
                logging.debug("tile %d is not completed yet"%t)
            elif s == 0:
                logging.debug("tile %d is not started yet"%t)
            else:
                logging.error("unknown result for tile %d: %s"%(t,str(s)))
            yield t

class DHDTTiles(Tiles):
    def __init__(self,cfg,reprocess=False):
        self._outputPrefix = cfg['grid']['output']
        self._deepstore = cfg['grid']['deepstore']
        Tiles.__init__(self,cfg['misc']['nproc'][0]*cfg['misc']['nproc'][1],self._outputPrefix+'_tiles.sqlite',reprocess=reprocess)

    def fname(self,i):
        return ncName(self._outputPrefix,n=i,deepstore=self._deepstore)
    
    def statusTile(self,i):
        if self.reprocess:
            return 0
        
        s = Tiles.statusTile(self,i)

        if s<2:
            # need to check netCDF file
            try:
                nc = netCDF4.Dataset(self.fname(i),"r",format="NETCDF4")
            except:
                # not started yet
                return 0

            if nc.completed == 1:
                # file says it is done
                done = 1.
                s=2
            else:
                # compute fraction done
                p = nc.variables['pixelCompleted']
                done = float(numpy.count_nonzero(p))/float(p.size)
                s=1

            self.updateTile(i,done=done)
        return s
            
if __name__ == '__main__':
    import argparse
    import dhdt
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--reprocess-data',action='store_true',default=False,help="process data even if previous run was successful")

    args = parser.parse_args()

    cfg = dhdt.Config()
    cfg.readCfg(args.config)

    tiles = DHDTTiles(cfg,reprocess=args.reprocess_data)
    for t in range(len(tiles)):
        print t,tiles.completed(t),tiles.isComplete(t)
    print tiles.totalCompleted()
    print len(tiles)

