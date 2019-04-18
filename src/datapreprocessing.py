# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from os import walk
import numpy as np
import h5py
import time
import math

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath = None, output_filepath = None):
    """ Runs data processing scripts to turn raw data from (../../data/raw) into
        cleaned data ready to be analyzed (saved in ../../data/processed).
        usage example: python datapreprocessing.py ../data/raw ../data/processed
    """
    logger = logging.getLogger(__name__)
    logger.info('preprocessing data set from raw data')
    sheetList, days = loadRaw(input_filepath)
    selectedareas = generate_forecast_areas()
    aggregateGridData(sheetList, days, selectedareas, '../data/interim')
    make_h5(outdir_ = output_filepath, sa = selectedareas)


def loadRaw(filepath = None):
    sheetList = []
    names = ['squareId', 'timeInterval', 'countryCode', 'smsIn', 'smsOut', 'callIn', 'callOut', 'Internet']
    dir_ = filepath
    for _, _, file in walk(dir_):
        days = len(file)
        for f in file:
            data = pd.read_table(dir_ + '/' + f, names=names)
            sheetList.append(data)
        print('Reading Finished. There are ' + str(len(sheetList)) + ' files in all.')
    return sheetList, days

def aggregateGridData(sheetList, days, selectedareas, output_filepath , header = None, index = None):
    # used to aggregate the data of each grid
    # the output file named 'grid#.csv', no headers and no index by default
    # csv format timeInterval,callin,callout,smsin,smsout,internet
    logger = logging.getLogger(__name__)
    gridNum = 10000
    for i in selectedareas:
        gridData = pd.DataFrame()
        for sheet in sheetList:
            tmp = sheet[sheet['squareId'] == i]
            gridData = pd.concat([gridData, tmp], axis=0).reset_index(drop=True)

        #aggregate data
        gd = gridData.groupby(['timeInterval']).sum()
        gd = gd.drop(['countryCode','squareId'],axis = 1)#.reset_index(drop = True)

        #deal with missing data
        gd = gd.fillna(0)


        '''not all the time are recorded in the dataset, so we need to check and insert those missing interval'''

        #check and insert missing timeInterval
        strTime = 1383260400000 # the beginning of timestamp of dataset, i.e. from 11.01 07:00
        interval = 600000
        tt = list(range(strTime, strTime + interval * 144 * days , interval))
        for id, time in enumerate(tt[:-1]):
            if not tt[id + 1] == time + interval:
                for missingInterval in range(time + interval, tt[id + 1],interval):
                    logging.warning('warning: missing time interval '+str(missingInterval)+'... and now inserting...')
                    gd.loc[missingInterval] = [0,0,0,0,0]

        gd = gd.sort_index()
        #write csv to file
        gd.to_csv(output_filepath +'/grid'+str(i)+'.csv',header = header)
        logging.info('grid'+str(i)+' aggregated\n')

def make_h5(indir_='../data/interim', outdir_='../data/processed',  sa = list(range(1,10001)),fname = 'demo_internet_data.h5'):
    #convert list of grid csv data into h5 format
    #indir_ refers to path to the csv files
    #fname is the name of file generated
    #temporal_gran is the temporal aggregating granduality
    #spatial_gran is the spatial aggregatin granduality
    #slots = 4320
    slots = len(pd.read_csv('{}/grid{}.csv'.format(indir_,sa[0]),header = None))
    rows  = 30#100
    cols = 30#100
    f = h5py.File(outdir_+'/demo_internet_data.h5','w')
    f2 = h5py.File(outdir_+'/demo_sms_data.h5'+fname,'w')
    f3 = h5py.File(outdir_+'/demo_call_data.h5'+fname,'w')

    f_time = f.create_dataset('date',(slots,),dtype = 'S13')
    f_data = f.create_dataset('data',(slots,1,rows,cols),dtype='float64')
    f2_time = f.create_dataset('date',(slots,),dtype = 'S13')
    f2_data = f.create_dataset('data',(slots,2,rows,cols),dtype='float64')
    f3_time = f.create_dataset('date',(slots,),dtype = 'S13')
    f3_data = f.create_dataset('data',(slots,2,rows,cols),dtype='float64')

    #generate time interval
    timeInter = [1383260400000+i*600000 for i in range(slots)]
    for i,t in enumerate(timeInter):
        f_time[i] = bytes(str(t).encode(encoding='utf-8'))
        f2_time[i] = bytes(str(t).encode(encoding='utf-8'))
        f3_time[i] = bytes(str(t).encode(encoding='utf-8'))
        
    for idg,grid in enumerate(sa):
        data = pd.read_csv('{}/grid{}.csv'.format(indir_,grid),names = ['timeInterval', 'smsIn', 'smsOut', 'callIn', 'callOut', 'Internet'])
        Row = math.ceil((1+idg)/rows)
        Row -= 1
        Col = cols if (1+idg) % cols == 0 else (1+idg) % cols
        Col -= 1
        #print(Row,Col)
        for i,t in enumerate(timeInter):
            #print(data[(data['timeInterval'] == t)].Internet)
            f_data[i,0,Row,Col] = float(data[(data['timeInterval'] == t)].Internet)
            f2_data[i,0,Row,Col] = float(data[(data['timeInterval'] == t)].smsIn)
            f2_data[i,1,Row,Col] = float(data[(data['timeInterval'] == t)].smsOut)
            f3_data[i,0,Row,Col] = float(data[(data['timeInterval'] == t)].callIn)
            f3_data[i,1,Row,Col] = float(data[(data['timeInterval'] == t)].callOut)
            #if i % slots == 100:
                #print(grid,i,0,Row,Col,f_data[i,0,Row,Col])
    f.close()
    f2.close()
    f3.close()
    
def generate_forecast_areas(sr = (40,70)):
    # by default the areas for prediction is [41:70,41:70]
    a = np.arange(1,10001).reshape(100,100)
    b = a[sr[0]:sr[1],sr[0]:sr[1]]
    return b.flatten().tolist()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
