import argparse
import pickle

import numpy as np
import s3fs
import pandas as pd
from metpy.io import Level2File

import time
import datetime
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as TPool

import tqdm
import logging


from RadarSlice_L2 import RadarSlice_L2
from RadarROI_L2 import RadarROI_L2

def parse_arg():
    """
    This function parses command line arguments to this script
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--var_a", type=int, default=1)
    parser.add_argument("--var_b", type=int, default=2)
    parser.add_argument("--save_name", type=str,required=True)
    parser.add_argument("-t", "--convTime", type=str, help="passed: -t <YYYYMMDDHHMMSS> Start time of observation (UTC)")
    parser.add_argument("-i", "--convInterval", type=str, default='0200', help="passed: -i <HHMM> Period of observation measured from start time (inclusive)")
    parser.add_argument("-d", "--convThreshMin", type=float, default='35.0', help="passed: -d Minimum threshold of reflectivity (dBZ) ")
    parser.add_argument("-clat", "--convLat", type=float, help="passed: -c <lat_float>; Lat of point of convection")
    parser.add_argument("-clon", "--convLon", type=float, help="passed: -c <lon_float>; Lon of point of convection")
    parser.add_argument("-b", "--convBearing", type=float, help="passed: -b Bearing of storm training, in Rads, measured CCW from East")
    parser.add_argument("-s", "--sensor", help=" 4 letter code for sensor")
    parser.add_argument("-sf", "--scaleFactor", type=float, default=1.0, help=" (Optional) Scale factor for ROI when performing sensitivity analysis")

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
    #print(aws_hourly_bucket)

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
            'KCYS':(41.15166, -104.80622)}

    offset = np.array([sensors[params['sensor']][0] - params['convLat'],
                        sensors[params['sensor']][1] - params['convLon']])

    baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),
                        (-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),
                        (0.8750,0.25,0.0,1.0)])     #crds of bounding box (Gridded degrees)

    roi.calc_cartesian()
    roi.shift_cart_orgin(offset=offset)

    #roi.extractROI(baseBearing=params['convBearing'])           # General locating
    roi.extractROI(baseCrds=baseCrds, baseBearing=params['convBearing'], scaleFactor=params['scaleFactor'])
    
    roi.clipped_axis_collapse(axis=0, mode="mean")

    reflectThresh = params['convThreshMin']                      # return strength threshold (135.0 = 35dbz)     
    roi.find_area(reflectThresh)
    roi.find_mean_reflectivity(reflectThresh)
    roi.find_variance_reflectivity(reflectThresh)
    d[roi.sweepDateTime] = [roi.sweepDateTime,roi.metadata,roi.sensorData,\
                            roi.mask,roi.xlocs,roi.ylocs,roi.clippedData,\
                            roi.polyVerts,offset,roi.area,roi.meanReflectivity,\
                            roi.varReflectivity, roi.clippedAxisCollapse]
    #del roi

if __name__ == "__main__":
    params = parse_arg()  # Parse command line arguments

    fs = s3fs.S3FileSystem(anon=True) # accessing all public buckets
    
    manager = mp.Manager()
    results = manager.dict()
    pool = TPool(12)
    jobs = []

    startDateTime = datetime.datetime.strptime(params['convTime'], '%Y%m%d%H%M')
    intervalDateTime = datetime.timedelta(hours = int(params['convInterval'][:2]), minutes=int(params['convInterval'][2:]))

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
                    intervalDateTime))]['L2File'].tolist()      
    logging.info(f'files: {[fs.info(obj)["Key"] for obj in filesToProcess]}')
    #if len(filesToProcess) < 8:
        #warnings.warn("n of radar inputs is not sufficent for curve smoothing",  UserWarning)
    print(f'Number of files to process: {len(filesToProcess)}')

# --- Stream files ahead of time to avoid error with multiprocessing and file handles ---
    filesToWorkers = []

    ### todo: read these in without streaming them
    start = time.time()
    for L2FileStream in filesToProcess:#tqdm(filesToProcess,desc="Streaming L2 Files"):
        try:
            if datetime.datetime.strptime(fs.info(L2FileStream)['Key'][39:53], '%Y%m%d_%H%M%S') >= datetime.datetime(2016, 1, 1):
                #filesToWorkers.append(fs.open(L2FileStream))
                filesToWorkers.append(Level2File(fs.open(L2FileStream)))
                #filesToWorkers.append(Level2File(L2FileStream.open()['Body']))
            else:
                print("uhh-oh, date prior to 2016-01-01, different read required from s3")
                #bytestream = BytesIO(L2FileStream.get()['Body'].read())
                #with gzip.open(bytestream, 'rb') as f:
                #    filesToWorkers.append(Level2File(f))  
        except:
            print("Value Error, Most likely in parsing header" )
    
    # --- Create pool for workers ---
    for file in filesToWorkers:
        job = pool.apply_async(calculate_radar_stats, (results, file))
        jobs.append(job)

    # --- Commit pool to workers ---
    for job in jobs:#tqdm(jobs,desc="Bounding & Searching Data"):
        job.get()

    pool.close()
    pool.join()

    columns =['sweepDateTime', 'metadata', 'sensorData', 'indices', 'xlocs', 'ylocs', 
                'data', 'polyVerts', 'offset', 'areaValue', 'refValue', 'varRefValue',
                'axisCollapseValues']
    print('Creating Dataframe... (This may take a while if plotting significant data)')
    resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)    #SUPER slow
    print('Converting datetimes...')
    resultsDF['sweepDateTime'] = pd.to_datetime(resultsDF.sweepDateTime)
    print('Sorting...')
    resultsDF.sort_values(by='sweepDateTime', inplace=True)
    #resultsDF.to_csv(params['output'] + '.csv', index = False)
    print(resultsDF[['areaValue','refValue']].head(5))
    print(resultsDF.info(verbose=True))

    print(f"collapse df vals: {resultsDF['axisCollapseValues'].values}")
    hov = np.array(resultsDF['axisCollapseValues'])#.squeeze()
    #print(np.shape(hov))


    elapsed = (f'{time.time() - start}')
    print(f'elapsed: {elapsed}')
    #print(filesToWorkers)

    # Write results to file with the provided "save_name"
    pickle.dump(elapsed, open('{}.p'.format(params['save_name']), 'wb'))