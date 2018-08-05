#!/bin/env python

__all__ = ['createOutDirs']

from grid import ncName, cfg2llgrid
import logging
import os.path, os
import math
import sys

def createDir(dname):
    if not os.path.exists(dname):
        os.makedirs(dname)


def createAllDirs(base,depth):
    if depth < 1:
        return
    for i in range(0,16):
        p = os.path.join(base,'{:x}'.format(i))
        createDir(p)
        createAllDirs(p,depth-1)

def createOutDirs(cfg):
    # create directory for data store
    dname = os.path.dirname(os.path.abspath(cfg['data']['store']))
    createDir(dname)

    if cfg['data']['storeType'] == 'multihdf':
        # create data store directory structure and meta data file
        dname = os.path.abspath(cfg['data']['store'])
        createDir(dname)
        mname = os.path.join(dname,'meta.json')

        llgrid = cfg2llgrid(cfg)
        if not os.path.exists(mname):
            llgrid.store(mname)

        # create the directories for the datastore
        storebase = os.path.join(dname,'multihdf')
        createDir(storebase)
        hexDB = '{:x}'.format(llgrid.ncells-1)
        createAllDirs(storebase,len(hexDB)-1)

    # create output directory
    outbase = os.path.dirname(os.path.abspath(cfg['grid']['output']))
    createDir(outbase)
        
    # create directory structure for tiles
    outbase = os.path.join(outbase,'tiles_%s'%os.path.basename(cfg['grid']['output']))
    createDir(outbase)
    maxTile = cfg['misc']['nproc'][0]*cfg['misc']['nproc'][1]-1
    hexTiles = '{:x}'.format(maxTile)
    createAllDirs(outbase,len(hexTiles)-1)

def main():
    import dhdt
    import argparse

    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog(),dhdt.batchParser()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    
    args = parser.parse_args()

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
        batch.serial(['createOutDirs',args.config])
    else:
        createOutDirs(cfg)
      

if __name__ == '__main__':
    main()
