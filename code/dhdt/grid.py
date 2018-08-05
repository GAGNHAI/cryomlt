"""handling the grid"""


__all__ = ['Grid','GridLonLat','cfg2grid','cfg2llgrid','checkNC','ncName']

from mpl_toolkits.basemap import Basemap
from matplotlib import path
import os.path
import numpy
import pyproj
import config
import netCDF4
import logging
from datetime import datetime
import fiona
from shapely.geometry import shape
from functools import partial
from shapely.ops import transform
import json

class GridNCVariable(object):
    """
    standard variable setup
    """
    def __init__(self,name,dtype=numpy.float32,dims=("time","y","x"),
                 fill_value=-1e6,coords="lon lat",units=None,
                 long_name = None, axis = None):
        self.name = name
        self.dtype = dtype
        self.dims = dims
        self.fill_value = fill_value
        self.coords = coords
        self.units = units
        self.long_name = long_name
        self.axis = axis
        
    def create(self,nc):
        v = nc.createVariable(self.name,self.dtype,self.dims,fill_value=self.fill_value)
        if self.coords is not None:
            v.coordinates = self.coords
        if self.units is not None:
            v.units = self.units
        if self.long_name is not None:
            v.long_name = self.long_name
        if self.axis is not None:
            v.axis = self.axis

    def check(self,nc):
        v = nc.variables[self.name]
        assert v.dtype == self.dtype
        assert v.dimensions == self.dims
        if self.fill_value is not None:
            assert v._FillValue == self.fill_value
        if self.coords is not None:
            assert v.coordinates == self.coords
        if self.units is not None:
            assert v.units == self.units
        if self.long_name is not None:
            assert v.long_name == self.long_name
        if self.axis is not None:
            assert v.axis == self.axis

class GridLonLat(object):
    """class holding a longitude/latitude grid"""
    def __init__(self,bbox,dlon=0.5,dlat=0.5):
        """
        bbox: the lower left and upper right corner of the region 
              [lllon,lllat,urlon,urlat]
        dlon: the EW size of each cell
        dlat: the SN size of each cell"""


        self._bbox = numpy.array(bbox,dtype=float)
        assert len(self._bbox)==4
        self._dlon = float(dlon)
        self._dlat = float(dlat)

        self._haveSouthPole = self._bbox[1] < -90+self.dlat
        self._haveNorthPole = self._bbox[3] >  90-self.dlat

        if self.haveSouthPole:
            self._bbox[1] = -90+self.dlat
        if self.haveNorthPole:
            self._bbox[3] = 90-self.dlat

        self._nlat = int(numpy.ceil((self._bbox[3]-self._bbox[1])/self.dlat))
        if self.haveSouthPole and self.haveNorthPole:
            self._dlat = (self._bbox[3]-self._bbox[1])/self.nlat
        elif self.haveNorthPole:
            self._bbox[1] = self._bbox[3] - self.nlat*self.dlat
        else:
            self._bbox[3] = self._bbox[1] + self.nlat*self.dlat

        assert self._bbox[0] < self._bbox[2]
        self._nlon = int(numpy.ceil((self._bbox[2]-self._bbox[0])/self.dlon))
        self._bbox[2] = self._bbox[0] + self.nlon*self.dlon

        self._ncells = self.nlat*self.nlon
        if self.haveSouthPole:
            self._ncells += 1
        if self.haveNorthPole:
            self._ncells += 1

        assert self._ncells > 0

    @property
    def haveSouthPole(self):
        return self._haveSouthPole
    @property
    def haveNorthPole(self):
        return self._haveNorthPole
        
    @property
    def dlon(self):
        return self._dlon
    @property
    def dlat(self):
        return self._dlat
    @property
    def nlon(self):
        return self._nlon
    @property
    def nlat(self):
        return self._nlat
    @property
    def ncells(self):
        return self._ncells

    @property
    def bbox(self):
        return self.getBBox()

    @property
    def meta(self):
        meta = {}
        meta['dlon'] = self.dlon
        meta['dlat'] = self.dlat
        meta['bbox'] = self.bbox
        meta['ncells'] = self.ncells
        return meta
    
    def getBBox(self,n=None):
        if n==None:
            if self.haveSouthPole:
                l0 = -90
            else:
                l0 = self._bbox[1]
            if self.haveNorthPole:
                l1 = 90.
            else:
                l1 = self._bbox[3]
            return [self._bbox[0],l0,self._bbox[2],l1]
        else:
            if self.haveSouthPole and n==0:
                return [self._bbox[0],-90.,self._bbox[2],-90.+self.dlat]
            if self.haveNorthPole and n==self.ncells-1:
                return [self._bbox[0],90.-self.dlat,self._bbox[2],90.]
            if self.haveSouthPole:
                # the first cell is the special and handled above
                n = n-1
            lat = n//self.nlon
            lon = n%self.nlon
            return [self._bbox[0]+lon*self.dlon,self._bbox[1]+lat*self.dlat,self._bbox[0]+(lon+1)*self.dlon,self._bbox[1]+(lat+1)*self.dlat]

    def getCells(self,bbox):
        # clip to bounding box of grid
        bb = [max(bbox[0],self.bbox[0]), max(bbox[1],self.bbox[1]), min(bbox[2],self.bbox[2]), min(bbox[3],self.bbox[3])]
        # check if South pole is included
        if bb[1] < -90+self.dlat:
            yield 0
            bb[1] = -90+self.dlat
        # check if North pole is included
        if bb[3] >= 90-self.dlat:
            yield self.ncells-1
            bb[3] = 90-self.dlat
        # loop over remaining grid
        lon_start = int((bb[0]-self.bbox[0])//self.dlon)
        lon_end   = min(int((bb[2]-self.bbox[0])//self.dlon)+1,self.nlon)
        lat_start = int((bb[1]-self.bbox[1])//self.dlat)
        lat_end   = min(int((bb[3]-self.bbox[1])//self.dlat)+1,self.nlat)
        for j in range(lat_start,lat_end):
            for i in range(lon_start,lon_end):
                yield j*self.nlon+i

    def store(self,fname):
        with open(fname,'w') as mfile:
            json.dump(self.meta,mfile,indent=4)

    def check(self,fname):
        ok = False    
        with open(fname,'r') as mfile:
            ok = True
            m = json.load(mfile)
            for k in self.meta:
                if k not in m:
                    ok = False
                else:
                    if k == 'bbox':
                        for i in range(4):
                            if abs(self.meta['bbox'][i]-m['bbox'][i])>1e-6:
                                ok = False
                    else:
                        if abs(self.meta[k]-m[k])>1e-6:
                            ok = False
        return ok
            
class Grid(object):
    """class holding the grid"""
    def __init__(self,bbox,inProj,outProj,posting=500.,tOrigin=datetime(1990,1,1),nproc=[1,1],mask=None,mask_buffer=20000.):
        """
        bbox: the lower left and upper right corner of the region 
              [lllon,lllat,urlon,urlat]
        inProj: a dictionary holding the proj4 data for the input grid,
                typically a latlon project
        outProj: a dictionary holding the proj4 data for the output grid
        posting: the spacing between the nodes of the output grid
        tOrigin: all times are in seconds since tOrigin
        nproc: a two element array/list containing the number of processes in 
               each spatial direction
        mask: name of a shapefile containing a mask
        mask_buffer: buffer around mask in meters
        """
        self._bbox = bbox
        self._inProj = inProj
        self._outProj = outProj
        self._tOrigin = tOrigin
        self._mask = []

        # compute lowerleft corner
        self._projection = pyproj.Proj(outProj)
        precision = 10
        pnts = []
        for c in [[0,1,2,1],[2,1,2,3],[2,3,0,3],[0,3,0,1]]:
            for ind in range(precision):
                lon = (precision-ind)*self._bbox[c[0]]/precision+ind*self._bbox[c[2]]/precision
                lat = (precision-ind)*self._bbox[c[1]]/precision+ind*self._bbox[c[3]]/precision
                pnts.append(self._projection(lon,lat))


        pnts = numpy.array(pnts)
        self._origin = numpy.array((pnts[:,0].min(),pnts[:,1].min()))
        ur = numpy.array((pnts[:,0].max(),pnts[:,1].max()))
        self._size = numpy.array((ur-self._origin)/posting,dtype=int)+1
        self._posting = posting

        logging.debug("ll %f %f"%(self.origin[0],self.origin[1]))
        logging.debug("ur %f %f"%(ur[0],ur[1]))
        logging.debug("size: %d %d"%(self.size[0],self.size[1]))

        # the number of processes are assumed to be on a 2D grid
        assert len(nproc) == 2
        self._nproc=numpy.array(nproc,dtype=int)

        if mask is not None:
            bb = shape({'type':'Polygon',
                        'coordinates': [[(self._bbox[0],self._bbox[1]),
                                         (self._bbox[0],self._bbox[3]),
                                         (self._bbox[2],self._bbox[3]),
                                         (self._bbox[2],self._bbox[1]),
                                         (self._bbox[0],self._bbox[1])]]})
            maskTransform = partial(
                pyproj.transform,
                pyproj.Proj(self._inProj),
                self.projection)
            with fiona.open(mask) as maskshp:
                for s in maskshp:
                    geo = shape(s['geometry'])
                    if geo.intersects(bb):
                        geo = transform(maskTransform,geo).buffer(mask_buffer)
                        geo = numpy.asarray(geo.exterior.coords).T
                        self._mask.append(geo)

    @property
    def tOrigin(self):
        return self._tOrigin

    @property
    def projection(self):
        """
        the projection object which can be used to transform between geographic
        and projected coordinates
        """
        return self._projection
        
    @property
    def origin(self):
        """the lower left corner of the projected coordinate system"""
        return self._origin

    @property
    def posting(self):
        """the node spacing of the projected coordinate system"""
        return self._posting

    @property
    def size(self):
        """the total number of nodes [nx,ny]"""
        return self._size

    def bounds(self,n):
        """start/end nodes for the nth process
        [[istart,iend],[jstart,jend]]
        """
        assert n<self.nproc[0]*self.nproc[1]
        s = self.size//self.nproc
        r = self.size%self.nproc

        ij = numpy.array([n%self.nproc[0],n//self.nproc[0]])

        iijj = []
        for i in range(2):
            start = min(ij[i],r[i])*(s[i]+1) + max(0,ij[i]-r[i])*s[i]
            end = start + s[i]
            if ij[i]<r[i]:
                # the first r[i] tiles get an extra node
                end = end + 1
            iijj.append([start,end])
        
        return iijj[0], iijj[1]

    def bboxXY(self,n=None,margin=0.):
        """the bounding box using projected coordinates of the nth process or if None the entire grid
        [llx,lly,urx,ury]
        set margin to a positive value (in meters) if you require extra points surrounding the bounding box. 
        """
        if n is None:
            return list(self.origin-margin)+list(self.origin+self.posting*self.size+margin)
        else:
            ii,jj = self.bounds(n)
            x = [self.origin[0]+ii[0]*self.posting,self.origin[0]+ii[1]*self.posting]
            y = [self.origin[1]+jj[0]*self.posting,self.origin[1]+jj[1]*self.posting]
            return [x[0]-margin,y[0]-margin,x[1]+margin,y[1]+margin]
            
    def bboxGeo(self,n=None,margin=0.):
        """the bounding box in geographic coordinates. If n is None then return the initial bounding box. Otherwise return the bounding box of the nth process. All four corners are transformed to geographic coordinates. The returned bounding box is then [min(long),min(lat),max(long),max(lat)]
        set margin to a positive value (in meters) if you require extra points surrounding the bounding box. 
        """
        if n is None:
            return self._bbox
        else:
            bb = self.bboxXY(n,margin=margin)
            pnts = []
            for c in [[0,1],[2,1],[2,3],[0,3]]:
                pnts.append(self.projection(bb[c[0]],bb[c[1]],inverse=True))
            pnts = numpy.array(pnts)
            return [min(pnts[:,0]),min(pnts[:,1]),max(pnts[:,0]),max(pnts[:,1])]
    
    @property
    def nproc(self):
        """the number of processors in x and y direction"""
        return self._nproc
    
    def mesh(self,n=None):
        """return the x,y coordinates of all nodes in the mesh"""

        if n is None:
            s = ([0,self.size[0]],[0,self.size[1]])
        else:
            s = self.bounds(n)


        x = self.origin[0]+numpy.arange(s[0][0],s[0][1])*self.posting
        y = self.origin[1]+numpy.arange(s[1][0],s[1][1])*self.posting

        return numpy.meshgrid(x,y)

    def mask(self,n=None):
        """return the mask"""

        xij,yij = self.mesh(n=n)
        shape = xij.shape

        xij = xij.reshape(shape[0]*shape[1])
        yij = yij.reshape(shape[0]*shape[1])

        if len(self._mask)>0:
            mask = numpy.zeros(shape[0]*shape[1],dtype=bool)

            for m in self._mask:
                m = path.Path(m.T).contains_points(numpy.array([xij,yij]).T)
                mask = mask | m
        else:
            mask = numpy.ones(shape[0]*shape[1],dtype=bool)
        
        
        mask = mask.reshape(shape)        
        return mask
        
    
    def plot(self,bbox=None):
        """plot the mesh on a map with bounding box bbox. If bbox is not use the bounding box of the grid"""
        if bbox is None:
            b = self._bbox
        else:
            b = bbox

        m = Basemap(llcrnrlon=b[0], llcrnrlat=b[1],
                    urcrnrlon=b[2], urcrnrlat=b[3],
                    resolution='l',projection=self._outProj['proj'],
                    lat_0=self._outProj['lat_0'],lon_0=self._outProj['lon_0'])
        m.drawcoastlines()
        m.drawparallels(numpy.arange(-80.,81.,5.))
        m.drawmeridians(numpy.arange(-180.,181.,5.))
        
        for i in range(self.nproc[0]*self.nproc[1]):
            xv,yv =  self.mesh(i)

            lon,lat = self.projection(xv.ravel(),yv.ravel(),inverse=True)
            x,y = m(lon,lat)
            m.plot(x,y,'.')

        for mask in self._mask:
            lon,lat = self.projection(mask[0,:],mask[1,:],inverse=True)
            x,y = m(lon,lat)
            m.plot(x,y)
        return m

    def netCDF(self,basename,n=None,store_tile_numbers=False,registration="node",overwrite=False,deepstore=True):
        timeUnits = "seconds since %s"%self.tOrigin.strftime("%Y-%m-%d %H:%M:%S")
        xv,yv = self.mesh(n)

        vars = [
            GridNCVariable('x',dims=("x",),axis='X',long_name = 'easting',units='m', coords=None, fill_value=None),
            GridNCVariable('y',dims=("y",),axis='Y',long_name = 'northing',units='m', coords=None, fill_value=None),
            GridNCVariable('time',dtype=numpy.float64,dims=("time",),units=timeUnits,coords=None, fill_value=None),
            GridNCVariable('lat',dims=("y","x"),long_name = "latitude", units = "degrees_north", coords=None, fill_value=None),
            GridNCVariable('lon',dims=("y","x"),long_name = "longitude", units = "degrees_east", coords=None, fill_value=None),
            GridNCVariable('topo',units="m"),
            GridNCVariable('rate',units="m/a"),
            GridNCVariable('stdResidual',fill_value=-1.),
            GridNCVariable('start',dtype=numpy.float64,fill_value=-1.,units=timeUnits),
            GridNCVariable('end',dtype=numpy.float64,fill_value=-1.,units=timeUnits),
            GridNCVariable('nMeasurements',dtype=numpy.int32,fill_value=-1),
            GridNCVariable('nMeasurementsUsed',dtype=numpy.int32,fill_value=-1),
            GridNCVariable('errorCovMat',fill_value=-1.),
            GridNCVariable('coeffLeadingEdgeWidth'),
            GridNCVariable('coeffX'),
            GridNCVariable('coeffY'),
            GridNCVariable('coeffX2'),
            GridNCVariable('coeffY2'),
            GridNCVariable('coeffXY'),
            GridNCVariable('coeffC'),
            GridNCVariable('coeffPower'),
            GridNCVariable('coeffPowerPoca'),
            GridNCVariable('pixelCompleted',dims=("y","x"),fill_value=-1,dtype=numpy.int8)
        ]
        if store_tile_numbers:
            vars.append(GridNCVariable('tile_number',dims=("y","x"),dtype=numpy.int32,fill_value=-1))
        
        fname = ncName(basename,n=n,deepstore=deepstore)

        newNC = True
        if os.path.exists(fname) and not overwrite:
            # check existing file
            try:
                nc = netCDF4.Dataset(fname,"a",format="NETCDF4")
                assert nc.registration == registration
                assert len(nc.dimensions['x']) == xv.shape[1]
                assert len(nc.dimensions['y']) == xv.shape[0]
                assert nc.projection == self.projection.srs
                if n is not None:
                    assert nc.tile == n

                for v in vars:
                    v.check(nc)
                newNC = False
            except:
                logging.warn("existing netCDF file %s exists but does not match, recreate it"%fname)
        if newNC:
            # create a new one        
            nc = netCDF4.Dataset(fname,"w",format="NETCDF4")
            nc.completed = 0
            nc.registration = registration
        
            nc.projection = self.projection.srs
            if n is not None:
                nc.tile = n
            xdim = nc.createDimension('x',xv.shape[1])
            ydim = nc.createDimension('y',xv.shape[0])
            tdim = nc.createDimension('time',None)

            for v in vars:
                v.create(nc)

            nc.variables['x'][:] = xv[0,:]
            nc.variables['y'][:] = yv[:,0]

            gx,gy = self.projection(xv,yv,inverse=True)
            nc.variables['lon'][:,:] = gx[:,:]
            nc.variables['lat'][:,:] = gy[:,:]

            nc.variables['pixelCompleted'][:,:] = numpy.where(self.mask(n=n),0,1)
            nc.variables['pixelCompleted'].flag_values = numpy.array([0,1,2,3],dtype=numpy.int8)
            nc.variables['pixelCompleted'].flag_meanings = "unprocessed masked_out nodata processed"
            if numpy.all(nc.variables['pixelCompleted'][:,:]>0):
                nc.completed = 1
            nc.sync()
                    
        return nc

def cfg2grid(cfg):
    assert isinstance(cfg,config.Config)

    grid = Grid(cfg['grid']['bbox'],
                cfg['data']['projection'],
                cfg['grid']['projection'],
                posting=cfg['grid']['posting'],
                nproc=cfg['misc']['nproc'],
                tOrigin=cfg['time']['timeOrigin'],
                mask = cfg.full_path(cfg['grid']['mask']),
                mask_buffer=cfg['grid']['mask_buffer'])
    return grid

def cfg2llgrid(cfg):
    assert isinstance(cfg,config.Config)

    llgrid = GridLonLat(cfg['data']['store_bbox'],
                        dlon=cfg['data']['store_dlon'],
                        dlat=cfg['data']['store_dlat'])
    return llgrid

def ncName(basename,n=None,deepstore=True):
    if n is not None:
        out = os.path.dirname(os.path.abspath(basename))
        if deepstore:
            path = '{:x}'.format(n)
            out = os.path.join(out,"tiles_%s"%os.path.basename(basename))
            for p in path[:-1]:
                out = os.path.join(out,p)
        fname = os.path.join(out,"%s_%06d.nc"%(os.path.basename(basename),n))
    else:
        fname = "%s.nc"%(basename)
    return fname

def checkNC(basename,n=None,deepstore=True):
    completed = False
    try:
        nc = netCDF4.Dataset(ncName(basename,n=n,deepstore=deepstore),"r")
        completed = nc.completed != 0
    except:
        pass
    return completed

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('config',metavar='CFG',help="name of the configuration file")
    parser.add_argument('--lower-left','-l',type=float,nargs=2,help="coordinates of the lower left corner for plotting")
    parser.add_argument('--upper-right','-u',type=float,nargs=2,help="coordinates of the upper right corner for plotting")
    parser.add_argument('--process','-p',metavar='N',type=int,help="compute tasks for process N")
    args = parser.parse_args()

    cfg = config.Config()
    cfg.readCfg(args.config)

    bbox = cfg['grid']['bbox']
    print bbox

    plot_bbox = []
    if args.lower_left is not None:
        plot_bbox += args.lower_left
    else:
        plot_bbox += bbox[:2]
    if args.upper_right is not None:
        plot_bbox += args.upper_right
    else:
        plot_bbox += bbox[2:]

    #plot_bbox = [-126.0004167,  71.9998167,-59.9997167,  84.0004167]
    grid = cfg2grid(cfg)
    m=grid.plot(plot_bbox)
    plt.show()

    nc = grid.netCDF(cfg['grid']['output'],args.process)
    nc.close()
