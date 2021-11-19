# NEXRAD Data Encapsulation - v0.1.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-04-28
#
# Updated
# 2021-10-16
#

# --- Imports ---
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import griddata
from Transformation_Matrix_2 import comp_matrix
from RadarSlice_L2 import RadarSlice_L2

class RadarROI_L2(RadarSlice_L2):
    
    @property
    def clippedData(self):
        if not hasattr(self, '_clippedData'):
            self._clippedData = np.empty((1))
        return self._clippedData

    @property
    def clippedRangeMap(self):
        if not hasattr(self, '_clippedRangeMap'):
            self._clippedRangeMap = np.empty((1))
        return self._clippedRangeMap

    @property
    def mask(self):
        return self._mask

    @property
    def area(self):
        if not hasattr(self, '_area'):
            self._area = -1.0
        return self._area

    @property
    def meanReflectivity(self):
        if not hasattr(self, '_meanReflectivity'):
            self._meanReflectivity = -1.0
        return self._meanReflectivity

    @property
    def varReflectivity(self):
        if not hasattr(self, '_varReflectivity'):
            self._varReflectivity = -1.0
        return self._varReflectivity

    @property
    def polyVerts(self):
        if not hasattr(self, '_polyVerts'):
            self._polyVerts = []
        return self._polyVerts
    
    @property
    def tm(self):
        if not hasattr(self,'_tm'):
            self._tm = comp_matrix(scale=np.ones(2), rotation=np.zeros(2), 
                                    shear=np.ones(2), translation=np.zeros(2))
        return self._tm

    @clippedData.setter
    def clippedData(self, value):
        self._clippedData = value

    @clippedRangeMap.setter
    def clippedRangeMap(self, value):
        self._clippedRangeMap = value

    @mask.setter
    def mask(self, value):
        self._mask = value

    @area.setter
    def area(self,value):
        self._area = value

    @meanReflectivity.setter
    def meanReflectivity(self, value):
        self._meanReflectivity = value

    @varReflectivity.setter
    def varReflectivity(self, value):
        self._varReflectivity = value

    @polyVerts.setter
    def polyVerts(self,value):
        self._polyVerts = value

    @tm.setter
    def tm(self, value):
        self._tm = value

    #Override
    def __init__(self, radarFile, sweep=0):
        super(RadarROI_L2, self).__init__(radarFile, sweep)

    def extractROI(self, baseCrds=None, baseBearing=0.0, scaleFactor=1.0):
        if baseCrds is None:
            baseCrds = np.array([(1.5,1.0,0.0,1.0),
                        (1.5,-1.0,0.0,1.0),
                        (-0.5,-1.0,0.0,1.0),
                        (-0.5,1.0,0.0,1.0),
                        (1.5,1.0,0.0,1.0)])    #default crds of bounding box (Gridded degrees)
        
        self.tm = comp_matrix(scale=np.ones(3)*scaleFactor, rotation=np.array([0,0, baseBearing]), 
                        shear=np.ones(3), translation=np.zeros(3))

        self.polyVerts = self.tm.dot(baseCrds.T).T[:,:2]    # Apply transformation Matrix, remove padding, and re-transpose

        # --- Generate ROI from coordiantes (above) create 2D boolean array to mask with ---
        xp,yp = self.xlocs.flatten(),self.ylocs.flatten()
        points = np.vstack((xp,yp)).T
        path = Path(self.polyVerts)
        grid = path.contains_points(points)
        grid = grid.reshape(np.shape(self.xlocs))
        rDataMasked = np.ma.masked_array(self.data, np.invert(grid))

        # --- Clip our masked array, create sub-array of data and rotate ---
        i, j = np.where(grid)
        self.mask = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')    # possible for there to be 0 (x or y) locs if crds are outside range of sweep for a certian sensor
        rDataMaskedClip = self.data[self.mask]
        rDataMaskClip = grid[self.mask]
        self.clippedData = rDataMaskedClip*rDataMaskClip

        rRangeMapMaskedClip = self.rangeMap[:,:][self.mask]     # self.rangeMap[:,:-1][self.mask] had to drop the tail clip for KMPX
        self.clippedRangeMap = rRangeMapMaskedClip*rDataMaskClip

        self.xlocs = self.xlocs[self.mask].astype(np.float16)
        self.ylocs = self.ylocs[self.mask].astype(np.float16)
        return self.clippedData

    #Override
    def find_area(self, reflectThresh=0.0):
        self.stackedData = np.dstack([self.clippedData, self.clippedRangeMap])                     # remove last range gate on the rangeMap
        self.area = np.nansum(np.where(self.stackedData[:,:,0]>= reflectThresh, self.stackedData[:,:,1], np.nan))
        return self.area

    #Override
    def find_mean_reflectivity(self, reflectThresh=0.0):
        if self.area == -1.0:
            find_area(reflectThresh)
        self.stackedData = np.dstack([self.clippedData, self.clippedRangeMap])
        self.meanReflectivity = np.nansum(np.where(self.stackedData[:,:,0]>= reflectThresh, self.stackedData[:,:,0]*self.stackedData[:,:,1], np.nan))/self.area #return product of reflectivity & weighting factor where >= thresh
        return self.meanReflectivity

    #Override
    def find_variance_reflectivity(self, reflectThresh=0.0):
        if self.area == -1.0:
            find_area(reflectThresh)
        self.stackedData = np.dstack([self.clippedData, self.clippedRangeMap])
        self.varReflectivity = np.nanvar(np.where(self.stackedData[:,:,0]>= reflectThresh, self.stackedData[:,:,0]*self.stackedData[:,:,1], np.nan))/self.area #return product of reflectivity & weighting factor where >= thresh
        #self.varReflectivity = np.var(np.array(list(filter(lambda x: x >= reflectThresh, self.clippedData.flatten()))))
        return self.varReflectivity

    #Override
    def get_interp_grid(self, reflectThresh = 0.0, grid_size_degree=0.0025):
        #lonMin, lonMax = np.min(self.xlocs), np.max(self.xlocs)
        #latMin, latMax = np.min(self.ylocs), np.max(self.ylocs)

        lonMin, lonMax = np.min(self.polyVerts[:,0]), np.max(self.polyVerts[:,0])
        latMin, latMax = np.min(self.polyVerts[:,1]), np.max(self.polyVerts[:,1])

        #Dimentions of regular grid
        latN = np.floor(np.abs(latMax-latMin)/grid_size_degree)
        lonN = np.floor(np.abs(lonMax-lonMin)/grid_size_degree)
        #ny, nx = 1500, 1500

        #Generate a regular grid to interpolate the data.
        xgrid, ygrid = np.mgrid[lonMin:lonMax:lonN*1j,
                                latMin:latMax:latN*1j]  # we use the "j" complex number notation here to make both endpoints inclusive for the np.mgrid function
        #reshape for interpolation
        interp_locs = np.dstack((self.xlocs, self.ylocs)).reshape(self.xlocs.size,2)
        grid = np.dstack((xgrid, ygrid)).reshape(xgrid.size,2)

        interp_grid= griddata(points=interp_locs,
              values=np.ravel(np.where(self.clippedData >= reflectThresh, self.clippedData, np.nan)),
              xi=grid, method='linear')

        interp_grid = interp_grid.reshape(np.shape(xgrid))

        return xgrid.T.astype(np.float16), ygrid.T.astype(np.float16), interp_grid.T.astype(np.float16)

    #Override
    def __str__(self):
        return ('string method for radarRoi')