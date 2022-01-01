# NEXRAD Data Encapsulation - v0.1.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-04-27
#
# Updated
# 2021-10-16
#

# --- Imports ---
import warnings
import operator

import numpy as np
import datetime

from Transformation_Matrix_2 import comp_matrix

class RadarSlice_L2(object):

    #Full list of properties avaliable at: https://unidata.github.io/MetPy/latest/api/generated/metpy.io.Level3File.html

    @property
    def metadata(self):
        if not hasattr(self,'_metadata'):
            self._metadata = None
            #warnings.warn("Radar_Slice: No input metadata parsed on __init__", UserWarning)
        return self._metadata
    
    @property
    def sweepDateTime(self):
        if not hasattr(self,'_sweepDateTime'):
            self._sweepDateTime = None
            warnings.warn("Radar_Slice: No input datetime parsed on __init__ , check input metadata",  UserWarning)
        return self._sweepDateTime

    @property
    def data(self):
        if not hasattr(self,'_data'):
            self._data = None
            warnings.warn("Radar_Slice: No input data parsed on __init__", UserWarning)
        return self._data

    @property
    def rangeMap(self):
        if not hasattr(self,'_rangeMap'):
            self._rangeMap = None
        return self._rangeMap

    @property
    def reflectivityUnits(self):
        if not hasattr(self, '_reflectivityUnits'):
            self._reflectivityUnits = '8Bit'
        return self._reflectivityUnits
    
    @property
    def sensorData(self):
        if not hasattr(self,'_sensorData'):
            self._sensorData = {'siteID' : ""}
            warnings.warn("Radar_Slice: No input sensor metadata parsed on __init__", UserWarning)
        return self._sensorData

    @property
    def mapData(self):
        if not hasattr(self,'_mapData'):
            self._mapData = None
            warnings.warn("Radar_Slice: No input sensor data mapping values parsed on __init__", UserWarning)
        return self._mapData
    
    @property
    def xlocs(self):
        return self._xlocs

    @property
    def ylocs(self):
        return self._ylocs

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

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @sweepDateTime.setter
    def sweepDateTime(self, value):
        self._sweepDateTime = value

    @data.setter
    def data(self, value):
        self._data = value

    @rangeMap.setter
    def rangeMap(self, value):
        self._rangeMap = value

    @reflectivityUnits.setter
    def reflectivityUnits(self, value):
        self._reflectivityUnits = value

    @sensorData.setter
    def sensorData(self, value):
        self._sensorData = value

    @mapData.setter
    def mapData(self, value):
        self._mapData = value

    @xlocs.setter
    def xlocs(self, value):
        self._xlocs = value

    @ylocs.setter
    def ylocs(self, value):
        self._ylocs = value

    @area.setter
    def area(self,value):
        self._area = value

    @meanReflectivity.setter
    def meanReflectivity(self,value):
        self._meanReflectivity = value

    @varReflectivity.setter
    def varReflectivity(self, value):
        self._varReflectivity = value

    # Constructor (init)
    def __init__(self, radarFile, sweep=0):
        f = radarFile

        # We use the first value to determine if the header is of type Msg31Hdr (2008 - pres) or Msg1Fmt (2007 & prior) 
        if type(f.sweeps[sweep][0][0][0]) is int:
            sweepDate = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(f.sweeps[sweep][0][0][1] - 1)
            self.sweepDateTime =  sweepDate + datetime.timedelta(milliseconds=f.sweeps[sweep][0][0][0])
        elif type(f.sweeps[sweep][0][0][0]) is bytes:
            self.sensorData = {'siteID' : str(f.sweeps[sweep][0][0][0])[2:-1]}
            sweepDate = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(f.sweeps[sweep][0][0][2] - 1)
            self.sweepDateTime =  sweepDate + datetime.timedelta(milliseconds=f.sweeps[sweep][0][0][1])

        self.az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])                    #create array of all azimuth angles within sweep
        
        if self.sweepDateTime >= datetime.datetime(2008, 1, 1):
            ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]                                          #get header of reflectivity info
            self.ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])
        elif self.sweepDateTime < datetime.datetime(2008, 1, 1):
            ref_hdr = f.sweeps[sweep][0][1]['REF'][0]
            self.ref = np.array([ray[1]['REF'][1] for ray in f.sweeps[sweep]]) 

        self.ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate          #generate ranges of gates from the first gate outwards

        self.data = np.ma.array(self.ref).astype(np.float16)
        self.data[np.isnan(self.data)] = np.ma.masked                                            # Mask out the zeros

        #range map - km^2 at each bin in range*azimuth
        rangeStep = ref_hdr.gate_width
        l = lambda x:(1.0/len(self.az))*((np.pi*(x+ref_hdr.gate_width)**2)-(np.pi*(x)**2))
        self.rangeMap = np.ones(tuple(map(operator.add,self.data[:, :-1].shape,(0,1)))) * \
                                np.array([[l(xi) for xi in self.ref_range]]).astype(np.float16)

    def calc_cartesian(self):
        # these values will be the same for both the data and for the rangeMap values
        self.kmPerDeg = 111.0
        self.xlocs = (self.ref_range * np.sin(np.deg2rad(self.az[0:, np.newaxis]))/self.kmPerDeg).astype(np.float16) # Convert az,range to x,y, change to deg from km
        self.ylocs = (self.ref_range * np.cos(np.deg2rad(self.az[0:, np.newaxis]))/self.kmPerDeg).astype(np.float16)
        return True

    def get_interp_grid(self, reflectThresh=0.0, grid_size_degree=0.025):
        lonMin, lonMax = np.min(self.xlocs), np.max(self.xlocs)
        latMin, latMax = np.min(self.ylocs), np.max(self.ylocs)

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

        interp_grid = griddata(points=interp_locs,
              values=np.ravel(np.where(self.clippedData >= reflectThresh, self.clippedData, np.nan)),
              xi=grid, method='linear')

        interp_grid = interp_grid.reshape(np.shape(xgrid))
        
        return xgrid, ygrid, interp_grid

    def shift_cart_orgin(self, offset):
        #offset must be in decimal degrees for (lat,lon)
        self.xlocs += offset[1]
        self.ylocs += offset[0]

        if self.ylocs.any() > 90 or self.ylocs.any() < -90:
            warnings.warn("Radar_Slice: offset (lat) produces values out of expected range [-90:90]",  UserWarning)
        if self.xlocs.any() > 180 or self.xlocs.any() < -180:
            warnings.warn("Radar_Slice: offset (lon) produces values out of expected range [-180:180]",  UserWarning)
        return True

    def mask_zeros(self):
        self.data[self.data == 0] = np.ma.masked                                            # convert 0s to masked
        return True

    def convert_dbz(self):
        if self.reflectivityUnits == '8Bit':
            self.data = (self.data-65)*0.5
            self.reflectivityUnits = 'dBZ'
        return True

    def find_area(self, reflectThresh=0.0):
        self.stackedData = np.dstack([self.data, self.rangeMap])                     # remove last range gate on the rangeMap
        #dep    #rsStackedData = self.stackedData.reshape((self.stackedData.shape[0])*(self.stackedData.shape[1]),2)  #Sholdn't need to rehape here, but we don't need the extra dim and would need to collapse it later
        #dep    #self.area = sum(map(lambda i: i >= reflectThresh, self.data.flatten()))
        #dep    #self.area = sum((rsStackedData * np.array(list(map(lambda i: i >= reflectThresh,rsStackedData)))).T[1]) # Grab the underlying cell area values
        self.area = np.nansum(np.where(self.stackedData[:,:,0]>= reflectThresh, self.stackedData[:,:,1], np.nan))
        return self.area

    def find_mean_reflectivity(self, reflectThresh=0.0):
        if self.area == -1.0:
            find_area(reflectThresh)
        self.stackedData = np.dstack([self.data, self.rangeMap])
        self.meanReflectivity = np.nansum(np.where(self.stackedData[:,:,0]>= reflectThresh, self.stackedData[:,:,0]*self.stackedData[:,:,1], np.nan))/self.area #return product of reflectivity & weighting factor where >= thresh
        return self.meanReflectivity

    def find_variance_reflectivity(self, reflectThresh=0.0):
        if self.area == -1.0:
            find_area(reflectThresh)
        self.stackedData = np.dstack([self.data, self.rangeMap])
        self.varReflectivity = np.nanvar(np.where(self.stackedData[:,:,0]>= reflectThresh, self.stackedData[:,:,0]*self.stackedData[:,:,1], np.nan))/self.area #return product of reflectivity & weighting factor where >= thresh
        #self.varReflectivity = np.var(np.array(list(filter(lambda x: x >= reflectThresh, self.clippedData.flatten()))))
        return self.varReflectivity

    def clear_data(self):
        self.data = None

    def __str__(self):
        return ('string method for radarSlice')