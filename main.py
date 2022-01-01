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
from multiprocessing.pool import Pool

import logging

import matplotlib.pyplot as plt 
from matplotlib import dates as mpl_dates

from RadarSlice_L2 import RadarSlice_L2
from RadarROI_L2 import RadarROI_L2

import warnings     # Debug (progress bars bugged by matplotlib futurewarnings output being annoying)
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_arg():
    '''
    This function parses command line arguments to this script
    '''
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

    parser.add_argument("-l", "--logLevel", dest="logLevel", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="(default: %(default)s) Set the logging level")


    params = vars(parser.parse_args())

    if params['logLevel']:
        logging.basicConfig(level=getattr(logging, params['logLevel']))

    return params

def pull_data(startDateTime, station):
    '''
    Pulls all radar data streams for a specified station (One hr incraments)
    Param:  <Datetime> timestamp of radar file 
            <String> station code
    Return: <List> of s3 bucket handles which contains L2 radar data for each hourly bucket
            <List> of Datetimes which correspond to each hourly bucket
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

def query_data(startDateTime, intervalDateTime, station, hrIter=datetime.timedelta(hours=0)):
    '''
    Pulls all radar file handles for a specified station (One hr increments)
    Param:  startDateTime <Datetime> from which to start 
            intervalDateTime <Datetime> period following startDateTime in which to query
            station <String> station code
            hrIter <TimeDelta> (optinal) offset from startDateTime
    Return: <List> of s3 bucket handles which contain L2 radar data for specified datetime interval
    '''

    # Query all L2 files for the sensor
    totalRadarObjects = []
    totalSweepDateTimes = []

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

    filesToWorkers = []

    ### todo: read these in without streaming them to local if prior to 2016
    for L2FileStream in filesToProcess:
        if str(L2FileStream)[-4:] =="_MDM":                 #drop off any *_MDM files
            continue
        try:
            if datetime.datetime.strptime(fs.info(L2FileStream)['Key'][39:53], '%Y%m%d_%H%M%S') >= datetime.datetime(2016, 1, 1):
                filesToWorkers.append(L2FileStream)
            else:
                print("uhh-oh, date prior to 2016-01-01, different read required from s3")
                bytestream = BytesIO(L2FileStream.get()['Body'].read())
                with gzip.open(bytestream, 'rb') as f:
                    filesToWorkers.append(Level2File(f))  
        except:
            print("Value Error, Most likely in parsing header" )

    print(f'Number of files to process: {len(filesToWorkers)}')
    return filesToWorkers

def calculate_radar_stats(d, radarFile):
    '''
    Driver for the conversion and calculation of any stats in the radar objects, run in parellel with multiprocessing
    Param:  d <dict> Output for multiprocessing
            radarfile <string>  Handle for L2 file 
    Return: None 
    '''
    L2File = Level2File(fs.open(radarFile))


    roi = RadarROI_L2(radarFile=L2File)

    # we **should** be pulling these data directly from the L2 file rather than this dict
    sensors = {'KMVX':(47.52806, -97.325), 'KBIS':(46.7825, -100.7572), 
            'KMBX':(48.3925, -100.86444), 'KABR':(45.4433, -98.4134), 
            'KFSD':(43.5778, -96.7539), 'KUDX':(44.125, -102.82944), 
            'KOAX':(41.32028, -96.36639), 'KLNX':(41.95778, -100.57583), 
            'KUEX':(40.32083, -98.44167), 'KGLD':(39.36722, -101.69333),
            'KCYS':(41.15166, -104.80622), 'KMPX':(44.848889, -93.565528) }

    offset = np.array([sensors[params['sensor']][0] - params['convLat'],
                        sensors[params['sensor']][1] - params['convLon']])

    '''
    baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),
                        (-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),
                        (0.8750,0.25,0.0,1.0)])     #crds of base bounding box (Gridded degrees)
    '''
    baseCrds = np.array([(1.00,1.00,0.0,1.0),(1.00,-1.00,0.0,1.0),
                    (-1.00,-1.00,0.0,1.0),(-1.00,1.00,0.0,1.0),
                    (1.00,1.00,0.0,1.0)])     #crds of base bounding box (Gridded degrees)

    # The Western Michigan University 'W'
    '''
    baseCrds = np.array([[-0.93601896, -0.48815166,  0.        ,  1.        ],
       [-0.6042654 , -0.48815166,  0.        ,  1.        ],
       [-0.6042654 , -0.30805687,  0.        ,  1.        ],
       [-0.66587678, -0.30805687,  0.        ,  1.        ],
       [-0.57345972,  0.0521327 ,  0.        ,  1.        ],
       [-0.44549763, -0.48815166,  0.        ,  1.        ],
       [-0.26303318, -0.48815166,  0.        ,  1.        ],
       [-0.13507109,  0.0521327 ,  0.        ,  1.        ],
       [-0.04265403, -0.30805687,  0.        ,  1.        ],
       [-0.10663507, -0.30805687,  0.        ,  1.        ],
       [-0.10663507, -0.48815166,  0.        ,  1.        ],
       [ 0.22748815, -0.48815166,  0.        ,  1.        ],
       [ 0.22748815, -0.30805687,  0.        ,  1.        ],
       [ 0.17772512, -0.30805687,  0.        ,  1.        ],
       [-0.03080569,  0.51184834,  0.        ,  1.        ],
       [-0.21800948,  0.51184834,  0.        ,  1.        ],
       [-0.35545024,  0.        ,  0.        ,  1.        ],
       [-0.492891  ,  0.51184834,  0.        ,  1.        ],
       [-0.67772512,  0.51184834,  0.        ,  1.        ],
       [-0.88625592, -0.30805687,  0.        ,  1.        ],
       [-0.93601896, -0.30805687,  0.        ,  1.        ],
       [-0.93601896, -0.48815166,  0.        ,  1.        ]])
    '''
    roi.calc_cartesian()
    roi.shift_cart_orgin(offset=offset)

    #roi.extractROI(baseBearing=params['convBearing'])           # General locating
    roi.extractROI(baseCrds=baseCrds, baseBearing=params['convBearing'], scaleFactor=params['scaleFactor'])    

    reflectThresh = params['convThreshMin']                      # return strength threshold (135.0 = 35dbz)     

    print('Entering interpolation')
    roiRegGrid = roi.get_interp_grid(reflectThresh=5.0, grid_size_degree=0.005)         # Interpolate a regular 2D Grid from the limits established by ROI polygon crds and a cell sixw
    roiAxisCollapse = np.nanmean(roiRegGrid[2], axis=0)          # Collapse that grid along the 0th axis to average at each unnique longitude spacing

    d[roi.sweepDateTime] = [roi.sweepDateTime,roi.metadata,roi.sensorData,\
                            roi.mask,roi.xlocs,roi.ylocs,roi.clippedData,\
                            roi.polyVerts,offset,roi.area,roi.meanReflectivity,\
                            roi.varReflectivity, roiRegGrid, roiAxisCollapse]
    del roi         #cleanup our large obj as fast as possible

if __name__ == "__main__":
    params = parse_arg()  # Parse command line arguments
    fs = s3fs.S3FileSystem(anon=True) # accessing all public buckets, need this with broad scope
    start = time.time()   # Just for elapsed time on program

    manager = mp.Manager()
    results = manager.dict()
    #pool = Pool(20)     #use if processing limited
    pool = TPool(10)   #use if I/O limited
    jobs = []

    # --- for a pervided datetime interval move through each
        # hourly bucket and pull all file handles that meet our
        # temporal query --- 
    startDateTime = datetime.datetime.strptime(params['convTime'], '%Y%m%d%H%M')
    intervalDateTime = datetime.timedelta(hours = int(params['convInterval'][:2]),
                                          minutes=int(params['convInterval'][2:]))
    filesToWorkers = query_data(startDateTime=startDateTime, 
                                 intervalDateTime=intervalDateTime,
                                 station=params['sensor'])
    
    # --- Create pool for workers ---
    for file in filesToWorkers:
        print(file)
        job = pool.apply_async(calculate_radar_stats, (results, file))
        jobs.append(job)

    # --- Commit pool to workers ---
    for job in jobs:
        job.get()

    pool.close()
    pool.join()
    del pool        #cleanup our large obj as fast as possible

    columns =['sweepDateTime', 'metadata', 'sensorData', 'indices', 'xlocs', 'ylocs', 
                'data', 'polyVerts', 'offset', 'areaValue', 'refValue', 'varRefValue',
                'regRectGrid','axisCollapseValues']
    
    print('Creating Dataframe... (This may take a while if plotting significant data)')
    resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)    #SUPER slow
    print('Converting datetimes...')
    resultsDF['sweepDateTime'] = pd.to_datetime(resultsDF.sweepDateTime)

    print('Sorting...')
    resultsDF.sort_values(by='sweepDateTime', inplace=True)

    elapsed = datetime.timedelta(seconds=(time.time() - start))
    print(type(elapsed))
    print(f"elapsed: {elapsed}")
    #.strftime('%H:%M:%S')

    plt.show()

    # Optionally save out calculated values dataframe as a .csv for injestion into another script
    # THIS WILL RESULT IN A LARGE FILE (Gbs for 24hrs of single station radar data) 
    # resultsDF.to_pickle('{}_raw_results_df.p'.format(params['save_name']))

    # Write results to file with the provided "save_name"

    result_out = np.dstack((*np.meshgrid(resultsDF['regRectGrid'].values[0][0][0,:],
                            np.datetime_as_string(resultsDF['sweepDateTime'].values, timezone='UTC', unit='s')),
                            np.stack(resultsDF['axisCollapseValues'].values)))

    pickle.dump(result_out, open('{}.p'.format(params['save_name']), 'wb'))