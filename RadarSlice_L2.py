# NEXRAD Data Encapsulation - v0.1.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-04-27
#
# Updated
# 2020-04-28
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
            warnings.warn("Radar_Slice: No input metadata parsed on __init__", UserWarning)
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

    #@property
    #def axisCollapse(self):
    #    if not hasattr(self, '_varReflectivity'):
    #        self.axisCollapse = None
    #    return self._axisCollapse

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

    #@axisCollapse.setter
    #def axisCollapse(self, value):
    #    self._axisCollapse = value

    # Constructor (init)
    def __init__(self, radarFile, sweep=0):
        f = radarFile
        #sweepDate = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(f.sweeps[sweep][0][0][2] - 1)
        #self.sweepDateTime =  sweepDate + datetime.timedelta(milliseconds=f.sweeps[sweep][0][0][1])
        
        
        #print(f'type: {type(f.sweeps[sweep][0][0])}')
        #print(f'value: {f.sweeps[sweep][0][0]}')

        # We use the first value to determine if the header is of type Msg31Hdr (2008 - pres) or Msg1Fmt (2007 & prior) 
        if type(f.sweeps[sweep][0][0][0]) is int:
            sweepDate = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(f.sweeps[sweep][0][0][1] - 1)
            self.sweepDateTime =  sweepDate + datetime.timedelta(milliseconds=f.sweeps[sweep][0][0][0])
        elif type(f.sweeps[sweep][0][0][0]) is bytes:
            self.sensorData = {'siteID' : str(f.sweeps[sweep][0][0][0])[2:-1]}
            sweepDate = datetime.datetime(1970,1,1,0,0) + datetime.timedelta(f.sweeps[sweep][0][0][2] - 1)
            self.sweepDateTime =  sweepDate + datetime.timedelta(milliseconds=f.sweeps[sweep][0][0][1])

        #print(f'SweepDT{self.sweepDateTime}')

        #print(f'site: {self.sensorData}')
        self.az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])                    #create array of all azimuth angles within sweep
        
        if self.sweepDateTime >= datetime.datetime(2008, 1, 1):
            ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]                                          #get header of reflectivity info
            self.ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])
        elif self.sweepDateTime < datetime.datetime(2008, 1, 1):
            ref_hdr = f.sweeps[sweep][0][1]['REF'][0]
            #print(f'ref: {f.sweeps[sweep][0][4]['REF'][1]}')
            self.ref = np.array([ray[1]['REF'][1] for ray in f.sweeps[sweep]]) 
        #print(f'ref: {self.ref[0]}')

        self.ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate          #generate ranges of gates from the first gate outwards

        self.data = np.ma.array(self.ref)
        self.data[np.isnan(self.data)] = np.ma.masked                                            # Mask out the zeros

        #range map - km^2 at each bin in range*azimuth
        rangeStep = ref_hdr.gate_width
        l = lambda x:(1.0/len(self.az))*((np.pi*(x+ref_hdr.gate_width)**2)-(np.pi*(x)**2))
        self.rangeMap = np.ones(tuple(map(operator.add,self.data[:, :-1].shape,(0,1)))) * \
                                np.array([[l(xi) for xi in self.ref_range]])

    def calc_cartesian(self):
        # these values will be the same for both the data and for the rangeMap values
        self.kmPerDeg = 111.0
        self.xlocs = (self.ref_range * np.sin(np.deg2rad(self.az[0:, np.newaxis]))/self.kmPerDeg) # Convert az,range to x,y, change to deg from km
        self.ylocs = (self.ref_range * np.cos(np.deg2rad(self.az[0:, np.newaxis]))/self.kmPerDeg)
        return True

    def __polar_to_cart_interpolation(lats, lons, data, grid_size_degree, mesh=False):
        # Page 32 of: https://skemman.is/bitstream/1946/16233/1/final_processingwithpython_dillon.pdf  
        # We want to interpolate it to a global x-degree grid
        deg2rad = np.pi/180.
        new_lats = np.linspace(grid_size_deg, 180, 180/grid_size_deg)
        new_lons = np.linspace(grid_size_deg, 360, 360/grid_size_deg)
        new_lats_mesh, new_lons_mesh = np.meshgrid(new_lats*deg2rad, new_lons*deg2rad)
        
        #We need to set up the interpolator object’’’
        lut = RectSphereBivariateSpline(lats*deg2rad, lons*deg2rad, data)
        
        #Finally we interpolate the data. The RectSphereBivariateSpline
        #object only takes 1-D arrays as input, therefore we need to do some reshaping.
        new_lats = new_lats_mesh.ravel()
        new_lons = new_lons_mesh.ravel()
        data_interp = lut.ev(new_lats,new_lons)
        if mesh == True:
            data_interp = data_interp.reshape((360/grid_size_deg,
                        180/grid_size_deg)).T
        return new_lats/deg2rad, new_lons/deg2rad, data_interp

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

    #def collpase_axis(self, axis=1, mode="mean"):
    #    if mode == "mean":
    #        self.axisCollapse = np.mean(self.data, axis=axis)
    #    elif mode == 'max':
    #        self.axisCollapse = np.max(self.data, axis=axis)
    #    elif mode == 'min':
    #        self.axisCollapse = np.min(self.data, axis=axis)
    #    else:
    #        print('axis collapse error')
    #    print(self.axisCollapse)
    #    return self.axisCollapse

    def __str__(self):
        return ('string method for radarSlice')