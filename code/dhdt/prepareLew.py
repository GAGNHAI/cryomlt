
import dhdt
import argparse

import numpy

def averageSpaceTime(p, name, spaceCorr=0, timeCorr = 0):
        
    Ndata = len(p['x'])
    param = numpy.empty(Ndata)
    param[:] = numpy.NAN
    
    if spaceCorr > 0:
        for indData in range(Ndata):
            
            timeRef = p['time'][indData]
            pnt = p['geometry'][indData]
            circle =  pnt.buffer(spaceCorr)
            selectedDataSpace = p[p['geometry'].within(circle)]
            
            if timeCorr > 0:
                index = (numpy.abs(selectedDataSpace['time']-timeRef)<=timeCorr)
                param[indData] = numpy.mean(selectedDataSpace[name][index])  
            else:
                param[indData] = numpy.mean(selectedDataSpace[name])
                
        p[name] = param
            
    
    return p
    
def linkPocaSwath(pPoca,pSwath,nameListPoca,nameListSwath):
    
    NNamePoca = len(nameListPoca)
    NNameSwath = len(nameListSwath)
    
    if NNamePoca != NNameSwath:
        raise Exception('nameListPoca and nameListSwath should contain the same number of entry')
    
    Ndata = len(pPoca[nameListPoca[0]])
    
    for indData in range(Ndata):
        timeRef = pPoca['startTime'][indData]
        wf_number = pPoca['wf_number'][indData]
        index = numpy.where( (pSwath['startTime']==timeRef) & (pSwath['wf_number']==wf_number) )

        if len(index[0])!=0:
            for indName in range(NNamePoca):
                namePoca = nameListPoca[indName]
                nameSwath = nameListSwath[indName]
                pSwath.loc[index[0],nameSwath] = pPoca.loc[indData,namePoca]
    
    return pSwath


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(parents=[dhdt.dhdtLog()])
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('-r','--reprocess-data',action='store_true',default=False,help="process data even if previous run was successful")
    parser.add_argument('--process','-p',metavar='N',default=0,type=int,help="compute tasks for process N")
    parser.add_argument('--tile-file','-T',metavar='TFILE',help="get tile IDs from file TFILE")
    parser.add_argument('--monitor-memory',action='store_true',default=False,help="monitor CPU and memory usage")
    args = parser.parse_args()
    dhdt.initLog(args)

    
    cfg = dhdt.Config()
    cfg.readCfg(args.config)
    
    grid = dhdt.cfg2grid(cfg)
    
    process = args.process
    
    radius = numpy.maximum(cfg['dhdt']['lew_where'],cfg['dhdt']['pow_where'])
    margin = cfg['dhdt']['radius']+radius
    
    print grid.bboxGeo(process,margin=margin)
    print grid.bboxGeo(process,margin=cfg['dhdt']['radius'])

    dataSwath,dataPoca = dhdt.getStore(cfg,mode='r')
    gPoca = dataPoca.getGeoPandas(crs=cfg['data']['projection'],bbox=grid.bboxGeo(process,margin=margin),
                              minPower=cfg['dhdt']['minPower'],spass=cfg['dhdt']['satellitePass'],
                              demDiffMadThresh=cfg['dhdt']['maxDemDiffMad'],demDiffThresh=cfg['dhdt']['maxDemDiff'])  
    pPoca = gPoca.to_crs(cfg['grid']['projection'])
    pPoca['x'] = pPoca.centroid.map(lambda pnt: pnt.x)
    pPoca['y'] = pPoca.centroid.map(lambda pnt: pnt.y)
    pPoca = pPoca.reset_index(drop=True)
    
    gSwath = dataSwath.getGeoPandas(crs=cfg['data']['projection'],bbox=grid.bboxGeo(process,margin=cfg['dhdt']['radius']),
                              minPower=cfg['dhdt']['minPower'],spass=cfg['dhdt']['satellitePass'],
                              demDiffMadThresh=cfg['dhdt']['maxDemDiffMad'],demDiffThresh=cfg['dhdt']['maxDemDiff']) 
    pSwath = gSwath.to_crs(cfg['grid']['projection'])
    pSwath['x'] = pSwath.centroid.map(lambda pnt: pnt.x)
    pSwath['y'] = pSwath.centroid.map(lambda pnt: pnt.y)
    pSwath= pSwath.reset_index(drop=True)
    

#    print pPoca[pPoca['time']>pPoca['time'][80][1]]

    
    print pPoca['leadEdgeW'][range(10)]
    print pPoca['power'][range(10)]
    pPoca = averageSpaceTime(pPoca, 'leadEdgeW', spaceCorr=5000, timeCorr = 2000)
    pPoca = averageSpaceTime(pPoca, 'power', spaceCorr=5000, timeCorr = 2000)
    print pPoca['leadEdgeW'][range(10)]
    print pPoca['power'][range(10)]
    print pPoca['wf_number'][range(10)]
     

    nameListPoca = ['power','leadEdgeW']
    nameListSwath = ['poca_power','leadEdgeW']
    pSwath = linkPocaSwath(pPoca,pSwath,nameListPoca,nameListSwath)
    print pSwath['wf_number'][380:408]
    print pSwath['leadEdgeW'][380:408]
    print pSwath['poca_power'][380:408]
    print '***Proportion Nan***'
    print len(pSwath['leadEdgeW'])
    
    nameList = ['leadEdgeW','poca_power']
    print pSwath.loc[380,'leadEdgeW']
    p_out = pSwath.dropna(axis = 0, how = 'any',subset = nameList )
    print len(p_out['leadEdgeW'])
    print p_out['leadEdgeW']

