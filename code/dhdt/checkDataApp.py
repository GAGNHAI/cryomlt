#!/bin/env python

import netCDF4
import logging
from tiles import DHDTTiles
import sys

def main():
    import dhdt
    import argparse
    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-t','--total',action='store_true',default=False,help="get total completed")
    parser.add_argument('-a','--all',action='store_true',default=False,help="display all tiles")
    parser.add_argument('-n','--no-update',action='store_true',default=False,help="do not update the database")
    parser.add_argument('-o','--output',metavar='TILELIST',help="write missing tile numbers to file TILELIST")
    
    args = parser.parse_args()
    dhdt.initLog(args)

    # read the configuration
    cfg = dhdt.Config()
    cfg.readCfg(args.config)

    if args.output is not None:
        out = open(args.output,'w')
    else:
        out = sys.stdout

    tiles = DHDTTiles(cfg)

    if args.all:
        ti = range(len(tiles))
    else:
        ti = iter(tiles)

    if args.no_update:
        ti = tiles.noUpdateIter()
       
    for t in ti:
        out.write('%d %f\n'%(t, tiles.completed(t)))        
        
    if args.total:
        out.write('total: %.2f%%\n'%(tiles.totalCompleted()*100.))
    
if __name__ == '__main__':
    main()
