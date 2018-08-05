__all__ = ['dhdtLog','initLog']

import logging
import argparse

def dhdtLog(add_help=False):
    logParser = argparse.ArgumentParser(add_help=add_help)
    logParser.add_argument('-l','--log-level',choices=['debug','info','warn','error'],default='info',help="set log level")
    logParser.add_argument('-L','--log-file',metavar='LOG',help='write logs to file LOG')
    return logParser

def initLog(args=None,level=logging.INFO,mpi_rank=None):
    logLevel = level
    if args is None:
        logFile=None
    else:
        if args.log_level == 'debug':
            logLevel = logging.DEBUG
        elif args.log_level == 'info':
            logLevel = logging.INFO
        elif args.log_level == 'warn':
            logLevel = logging.WARN
        elif args.log_level == 'error':
            logLevel = logging.ERROR
        logFile=args.log_file

    if logFile is not None and mpi_rank is not None:
        logFile = logFile+'_%06d'%mpi_rank
        
    logging.basicConfig(filename=logFile, level=logLevel,
                        format='%(asctime)s %(levelname)s:%(message)s')
    


if __name__ == '__main__':
    parser=dhdtLog(add_help=True)
    args = parser.parse_args()
    initLog(args)
    logging.info('do some logging')
