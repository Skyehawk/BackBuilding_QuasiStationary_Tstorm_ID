import argparse
import pickle

import numpy as np
import s3fs
import pandas as pd
from metpy.io import Level2File
from metpy.plots import add_timestamp, colortables, ctables
from io import BytesIO
import gzip

import time
import datetime
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as TPool

import logging

import matplotlib.pyplot as plt 
from matplotlib import dates as mpl_dates

from RadarSlice_L2 import RadarSlice_L2
from RadarROI_L2 import RadarROI_L2

def parse_arg():
    """
    This function parses command line arguments to this script
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_name", type=str,required=True)
    parser.add_argument("-t", "--convTime", type=str, help="passed: -t <YYYYMMDDHHMMSS> Start time of observation (UTC)")
    parser.add_argument("-i", "--convInterval", type=str, default='0200', help="passed: -i <HHMM> Period of observation measured from start time (inclusive)")
    parser.add_argument("-d", "--convThreshMin", type=float, default='35.0', help="passed: -d Minimum threshold of reflectivity (dBZ) ")
    parser.add_argument("-clat", "--convLat", type=float, help="passed: -c <lat_float>; Lat of point of convection")
    parser.add_argument("-clon", "--convLon", type=float, help="passed: -c <lon_float>; Lon of point of convection")
    parser.add_argument("-b", "--convBearing", type=float, help="passed: -b Bearing of storm training, in Rads, measured CCW from East")
    parser.add_argument("-s", "--sensor", help=" 4 letter code for sensor")
    parser.add_argument("-sf", "--scaleFactor", type=float, default=1.0, help=" (Optional) Scale factor for ROI when performing sensitivity analysis")
    parser.add_argument("-th", "--thinning", type=int, default=2, help=" (Optional) thinning of NEXRAD files to process")

    params = vars(parser.parse_args())

    return params

def pull_data(startDateTime, station):
    '''
    Pulls all radar data streams for a specified station (One hr incraments)
    Param:  <Datetime> from which to start 
            <String> station code
    Return: <List> of s3 bucket handles which contains L2 radar data
            <List> of Datetimes which correspond to each bucket
    '''
    dt = startDateTime
    fs = s3fs.S3FileSystem(anon=True) # accessing all public buckets

    aws_hourly_bucket = fs.glob(f'noaa-nexrad-level2/{dt:%Y}/{dt:%m}/{dt:%d}/{station}/{station}{dt:%Y%m%d_%H}*')

    objects = []
    sweepDateTimes = []
    for obj in aws_hourly_bucket:
        objects.append(obj)
        sweepDateTimes.append(datetime.datetime.strptime(fs.info(obj)['Key'][39:53], '%Y%m%d_%H%M%S'))
    return objects, sweepDateTimes

def calculate_radar_stats(d, radarFile):
    '''
    Driver for the conversion and calculation of any stats in the radar objects, run in parellel with multiprocessing
    Param:  d <dict> Output for multiprocessing
            radarfile <metpy.io Level2File> 
    Return: None 
    '''
    roi = RadarROI_L2(radarFile=radarFile)

    sensors = {'KMVX':(47.52806, -97.325), 'KBIS':(46.7825, -100.7572), 
            'KMBX':(48.3925, -100.86444), 'KABR':(45.4433, -98.4134), 
            'KFSD':(43.5778, -96.7539), 'KUDX':(44.125, -102.82944), 
            'KOAX':(41.32028, -96.36639), 'KLNX':(41.95778, -100.57583), 
            'KUEX':(40.32083, -98.44167), 'KGLD':(39.36722, -101.69333),
            'KCYS':(41.15166, -104.80622), 'KMPX':(44.848889, -93.565528) }

    offset = np.array([sensors[params['sensor']][0] - params['convLat'],
                        sensors[params['sensor']][1] - params['convLon']])

    #baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),
    #                    (-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),
    #                    (0.8750,0.25,0.0,1.0)])     #crds of base bounding box (Gridded degrees)

    baseCrds = np.array([(1.00,2.00,0.0,1.0),(1.00,-2.00,0.0,1.0),
                    (-1.00,-2.00,0.0,1.0),(-1.00,2.00,0.0,1.0),
                    (1.00,2.00,0.0,1.0)])     #crds of base bounding box (Gridded degrees)

    roi.calc_cartesian()
    roi.shift_cart_orgin(offset=offset)

    #roi.extractROI(baseBearing=params['convBearing'])           # General locating
    roi.extractROI(baseCrds=baseCrds, baseBearing=params['convBearing'], scaleFactor=params['scaleFactor'])    

    reflectThresh = params['convThreshMin']                      # return strength threshold (135.0 = 35dbz)     
    #roi.find_area(reflectThresh)
    #roi.find_mean_reflectivity(reflectThresh)
    #roi.find_variance_reflectivity(reflectThresh)

    print('Entering interpolation')
    roiRegGrid = roi.get_interp_grid(reflectThresh=5.0, grid_size_degree=0.01)         # Interpolate a regular 2D Grid from the limits established by ROI polygon crds and a cell sixw
    roiAxisCollapse = np.nanmean(roiRegGrid[2], axis=0)          # Collapse that grid along the 0th axis to average at each unnique longitude spacing

    d[roi.sweepDateTime] = [roi.sweepDateTime,roi.metadata,roi.sensorData,\
                            roi.mask,roi.xlocs,roi.ylocs,roi.clippedData,\
                            roi.polyVerts,offset,roi.area,roi.meanReflectivity,\
                            roi.varReflectivity, roiRegGrid, roiAxisCollapse]
    del roi

if __name__ == "__main__":
    params = parse_arg()  # Parse command line arguments

    fs = s3fs.S3FileSystem(anon=True) # accessing all public buckets
    
    manager = mp.Manager()
    results = manager.dict()
    pool = TPool(12)
    jobs = []

    startDateTime = datetime.datetime.strptime(params['convTime'], '%Y%m%d%H%M')
    intervalDateTime = datetime.timedelta(hours = int(params['convInterval'][:2]),
                                          minutes=int(params['convInterval'][2:]))

    station = params['sensor']

    hrIter = datetime.timedelta(hours=0)
    
    # Query all L2 files for the sensor
    totalRadarObjects = []
    totalSweepDateTimes = []
    hrIter = datetime.timedelta(hours=0)
    while True:                                                                                 # grab a specific interval of files
        radarObjects, sweepDateTimes = pull_data(startDateTime=(startDateTime+hrIter),\
                                                 station=station)
        totalRadarObjects.extend(radarObjects[:-1])
        totalSweepDateTimes.extend(sweepDateTimes[:-1])                                     # remove trailing *_MDM file
        if totalSweepDateTimes[-1] - startDateTime >= intervalDateTime:
            break
        else: 
            hrIter += datetime.timedelta(hours=1)
    fileDict = {'L2File':totalRadarObjects, 'Time':totalSweepDateTimes}
    fileDF = pd.DataFrame(fileDict)

    #print(f'Station: {station}, Start time: {startDateTime}, Interval: {intervalDateTime}, End Time: {startDateTime + intervalDateTime}')
    
    filesToProcess = fileDF[((fileDF['Time'] >= startDateTime) \
                    & (fileDF['Time'] <= startDateTime + \
                    intervalDateTime))]['L2File'].tolist()[::params['thinning']]
    logging.info(f'files: {[fs.info(obj)["Key"] for obj in filesToProcess]}')
    print(f'Number of files to process: {len(filesToProcess)}')

# --- Stream files ahead of time to avoid error with multiprocessing and file handles ---
    filesToWorkers = []

    ### todo: read these in without streaming them to local if prior to 2016
    start = time.time()
    for L2FileStream in filesToProcess:#tqdm(filesToProcess,desc="Streaming L2 Files"):
        try:
            if datetime.datetime.strptime(fs.info(L2FileStream)['Key'][39:53], '%Y%m%d_%H%M%S') >= datetime.datetime(2016, 1, 1):
                filesToWorkers.append(Level2File(fs.open(L2FileStream)))
            else:
                print("uhh-oh, date prior to 2016-01-01, different read required from s3")
                bytestream = BytesIO(L2FileStream.get()['Body'].read())
                with gzip.open(bytestream, 'rb') as f:
                    filesToWorkers.append(Level2File(f))  
        except:
            print("Value Error, Most likely in parsing header" )
    
    # --- Create pool for workers ---
    for file in filesToWorkers:
        job = pool.apply_async(calculate_radar_stats, (results, file))
        jobs.append(job)

    # --- Commit pool to workers ---
    for job in jobs:
        job.get()

    pool.close()
    pool.join()

    columns =['sweepDateTime', 'metadata', 'sensorData', 'indices', 'xlocs', 'ylocs', 
                'data', 'polyVerts', 'offset', 'areaValue', 'refValue', 'varRefValue',
                'regRectGrid','axisCollapseValues']
    
    print('Creating Dataframe... (This may take a while if plotting significant data)')
    resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)    #SUPER slow
    print('Converting datetimes...')
    resultsDF['sweepDateTime'] = pd.to_datetime(resultsDF.sweepDateTime)
    
    print('Sorting...')
    resultsDF.sort_values(by='sweepDateTime', inplace=True)
    print(resultsDF[['areaValue','refValue']].head(5))
    print(resultsDF.info(verbose=True))

    elapsed = (f'{time.time() - start}')
    print(f'elapsed: {elapsed}')

    
    fig, axes = plt.subplots(2,2, figsize=(20,20),
        gridspec_kw={'width_ratios':[10, 10], 
     'height_ratios': [10, 10], 'wspace': 0.375,
     'hspace': 0.375})

    date_format = mpl_dates.DateFormatter('%H:%Mz')
    norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', 5,5)


    print(np.shape(resultsDF['regRectGrid'].values[0][0][0,:]))
    print(np.shape(np.stack(resultsDF['axisCollapseValues'].values)))

    axes[0][0].pcolormesh(
    *np.meshgrid(resultsDF['regRectGrid'].values[0][0][0,:],
     resultsDF['sweepDateTime'].values),
     np.stack(resultsDF['axisCollapseValues'].values), 
     norm=norm, cmap=cmap, shading='auto')
    
    axes[0][0].set_xlabel("Degrees Longitude From Initiation Point")
    axes[0][0].set_ylabel("Time (UTC)")
    axes[0][0].set_title("Hovmöller Diagram - Mean Reflectivity")

    plt.show()

    '''
    # --- Plot time series---
    fig, axes = plt.subplots(4, 4, figsize=(30, 30),
     gridspec_kw={'width_ratios': [10, 10, 10, 10], 
     'height_ratios': [10, 1, 10, 1], 'wspace': 0.375,
     'hspace': 0.375})

    date_format = mpl_dates.DateFormatter('%H:%Mz')

    for i, (dt, record) in enumerate(resultsDF.iterrows()):
        plotx = i%4
        ploty = int(i/4)

        negXLim = -.5
        posXLim = 1.5
        negYLim = -1.0
        posYLim = 1.0


        norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', 5,5)
        tempdata = record['regRectGrid'][2]                  # create a deep copy of data to maipulate for plotting
        tempdata[tempdata == 0] = np.ma.masked                      # mask out 0s for plotting
 
        axes[ploty][plotx].pcolormesh(record['regRectGrid'][0], record['regRectGrid'][1], 
                                        tempdata, norm=norm, cmap=cmap, shading='auto')        

        axes[ploty][plotx].set_aspect(aspect='equal')
        axes[ploty][plotx].set_xlim(negXLim, posXLim)
        axes[ploty][plotx].set_ylim(negYLim, posYLim)
        #axes[ploty+1][plotx].set_ylim(negYLim, posYLim)
        pVXs, pVYs = zip(*record['polyVerts'])                      # create lists of x and y values for transformed polyVerts
        axes[ploty][plotx].plot(pVXs,pVYs)
        if negXLim < record['offset'][1] < posXLim and \
            negYLim < record['offset'][0] < posYLim: 
            axes[ploty][plotx].plot(record['offset'][1], record['offset'][0], 'o')          # Location of the radar
            axes[ploty][plotx].text(record['offset'][1], record['offset'][0], record['sensorData']['siteID'])
            
        axes[ploty][plotx].plot(0.0, 0.0, 'bx')                     # Location of the convection
        axes[ploty][plotx].text(0.0, 0.0, str(params['convLat']) + ' , ' + str(params['convLon']))
        #add_timestamp(axes[ploty][plotx], record['sweepDateTime'], y=0.02, high_contrast=True)
        axes[ploty][plotx].tick_params(axis='both', which='both')

        # Create an axis average and add to plot
        ax0_mean_a = np.nanmean(record['regRectGrid'][2], axis=0)
        ax0_mean = np.tile(ax0_mean_a,(2,1))                        # Tile our data so we have a 2d array for pcolormesh(i.e. [1,3,2] --> [[1,3,2],[1,3,2]])
        ax0_Xs_a = record['regRectGrid'][0][0,:]
        ax0_Ys_a = np.linspace(0,1,2)
        ax0_Xs, ax0_Ys = np.meshgrid(ax0_Xs_a, ax0_Ys_a)            # Create a mesh-grid for the above tiled data
        axes[ploty + 1][plotx].pcolormesh(ax0_Xs.T, ax0_Ys.T, ax0_mean.T, norm=norm, cmap=cmap, shading='auto')
        axes[ploty + 1][plotx].set_yticks([])
        axes[ploty+1][plotx].set_xlim(negXLim, posXLim)

# ------ debug ----
        tempdata2 = record['data']
        tempdata2[tempdata2 == 0] = np.ma.masked                      # mask out 0s for plotting

        axes[ploty + 2][plotx].pcolormesh(record['xlocs'], record['ylocs'], 
                                        tempdata2, norm=norm, cmap=cmap, shading='auto')        
        axes[ploty + 2][plotx].set_aspect(aspect='equal')
        axes[ploty + 2][plotx].set_xlim(negXLim, posXLim)
        axes[ploty + 2][plotx].set_ylim(negYLim, posYLim)
        pVXs, pVYs = zip(*record['polyVerts'])                      # create lists of x and y values for transformed polyVerts
        axes[ploty + 2][plotx].plot(pVXs,pVYs)
        if negXLim < record['offset'][1] < posXLim and \
            negYLim < record['offset'][0] < posYLim: 
            axes[ploty + 2][plotx].plot(record['offset'][1], record['offset'][0], 'o')          # Location of the radar
            axes[ploty + 2][plotx].text(record['offset'][1], record['offset'][0], record['sensorData']['siteID'])
            
        axes[ploty + 2][plotx].plot(0.0, 0.0, 'bx')                     # Location of the convection
        axes[ploty + 2][plotx].text(0.0, 0.0, str(params['convLat']) + ' , ' + str(params['convLat']))
        #add_timestamp(axes[ploty + 2][plotx], record['sweepDateTime'], y=0.02, high_contrast=True)
        axes[ploty + 2][plotx].tick_params(axis='both', which='both')



# ------- debug ----



    plt.show()
    '''

    # Write results to file with the provided "save_name"
    pickle.dump(elapsed, open('{}.p'.format(params['save_name']), 'wb'))