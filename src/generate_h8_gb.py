#!/usr/bin/env python
"""
Fusiun - Multi-Satellite Fire Detection
Generates hotspot composite products based on hotspot detections from
multiple satellites.

@author: Wong Songhan <Wong_Songhan@nea.gov.sg>
@date: 9 Jun 2020
@version: 2.0

python generate_fusiun.py --start-date "2019-07-01 00:00" --end-date "2019-07-01 08:30"

TODO:
1. Filter coastline AHI hotspots using buffered .shp files

"""
# import matplotlib
# matplotlib.use('Agg')

import logging
import logging.config
import argparse
import os
import json
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy import crs as ccrs
from asmclib import geohotspot
from datetime import datetime, timedelta
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shapely.geometry import box


def main():

    logging.config.fileConfig(fname=os.path.join('config', 'log.config'), disable_existing_loggers=False)

    # Get the logger specified in the file
    f_handler = logging.FileHandler(os.path.join('logs', 'generate_h8_gb.log'))
    f_handler.setLevel(logging.DEBUG)
    log = logging.getLogger(__name__)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    log.addHandler(f_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prob-minimum', default=0.4, dest='prob_minimum',
                        help='Minimum probability to be considered as fire grids')
    parser.add_argument('-s', '--start-date', dest='start_date_str', help='Start date')
    parser.add_argument('-e', '--end-date', dest='end_date_str', help='End date')
    parser.add_argument('--ahi-hotspot-folder', dest='ahi_folder', default=os.path.join('..', 'data', 'raw',
                        'hotspots', 'ahi'), help='AHI hotspot folder')
    parser.add_argument('-n', '--name', dest="prefix_name", default='FUSIUN_NRT_2km_', help="Prefix for output file names")

    args = parser.parse_args()
    log.debug(args)
    prob_minimum = args.prob_minimum

    with open(os.path.join('config', 'config.json'), "r") as read_file:
        json_config = json.load(read_file)

    clipping_box = json_config['parameters']['clipping_box']
    sat_resolution_meter = json_config['sat_resolution_meter']
    shapefile_path = json_config['shapefile']['path']
    bounding_box = json_config['plotting']['bounding_box']
    dpi = json_config['plotting']['dpi']
    fusiun_ml_model_fpath = json_config['fusiun_ml_model']['path']
    fusiun_predict_features = json_config['fusiun_ml_model']['predict_features']
    h8_ml_model_fpath = json_config['h8_ml_model']['path']
    h8_predict_features = json_config['h8_ml_model']['predict_features']
    low_prob_thres = json_config['parameters']['low_prob_thres']
    med_prob_thres = json_config['parameters']['med_prob_thres']


    #read in grid shapefile
    try:
        df_grid = geopandas.read_file(args.grid_shp)
        df_grid.crs = {'init': 'epsg:3857'}
        log.debug(args.grid_shp + ' loaded successfully!')
    except Exception as e:
        log.error(args.grid_shp + ' cannot be loaded !')
        exit()

    start_date = datetime.strptime(args.start_date_str, "%Y-%m-%d %H:%M")
    date_process = start_date
    end_date = datetime.strptime(args.end_date_str, "%Y-%m-%d %H:%M")

    geo_hs = geohotspot.GeoHotspot()

    log.info('Reading hotspot .txt files')

    while date_process <= (end_date + timedelta(days=1)):
        h8_files = os.path.join(args.ahi_folder, "H08_*" + date_process.strftime('%Y%m%d_')
                                + "*_L2WLFbet_FLDK.06001_06001.csv")
        geo_hs.parse_jaxa_hotspot_txt(file_path=h8_files)

        jp1_files = os.path.join(args.viirs_folder, date_process.strftime('%Y%m%d') + "*JP1*.txt")
        geo_hs.parse_viirs_afedr_txt(file_path=jp1_files, sat_name='NOAA20')

        npp_files = os.path.join(args.viirs_folder, date_process.strftime('%Y%m%d') + "*NPP*.txt")
        geo_hs.parse_viirs_afedr_txt(file_path=npp_files, sat_name='NPP')

        modis_files = os.path.join(args.modis_folder, "*14." + date_process.strftime('%y%j') + "*.txt")
        geo_hs.parse_modis_mod14_txt(file_path=modis_files)

        date_process = date_process + timedelta(days=1)

    # remove hotspots outside of clipping area
    geo_hs.clip_hotspot(clipping_box)
    # reject hotspots due to sun glint
    #geo_hs.reject_sunglint_hs('Himawari-8/9', max_sunglint_angle)

    geo_df = geo_hs.hs_df.copy()
    geo_df['aqua_weight'] = 0.0
    geo_df['terra_weight'] = 0.0
    geo_df['n20_weight'] = 0.0
    geo_df['npp_weight'] = 0.0
    geo_df['geo_weight'] = 0.0
    geo_df['confidence'] = geo_df['confidence'].fillna(0)
    geo_df.astype({'geo_weight': 'float64', 'confidence': 'float64'})

    geo_df['date'] = pd.to_datetime(geo_df['date'], format="%d/%m/%Y %H:%M:%S")
    # selects period of interest
    geo_df = geo_df[(geo_df['date'] >= start_date) & (geo_df['date'] <= end_date)]
    log.debug(geo_df['date'].unique())
    log.debug(geo_df[['satellite', 'date']].groupby(['satellite']).count())

    try:
        h8_ml_model = load(h8_ml_model_fpath)
        log.debug('Loaded trained H8 ML model from ' + h8_ml_model_fpath)
        log.debug(f'Model pipeline: {h8_ml_model}')
        geo_df.loc[geo_df['satellite'] == 'Himawari-8/9', 'geo_weight'] = h8_ml_model.predict_proba(
            geo_df.loc[geo_df['satellite'] == 'Himawari-8/9', h8_predict_features])[:, 1]
        log.info('Added in probabilities using H8 Gradient Boosting Model.')
    except Exception as e:
        log.exception(e)

    geo_df.loc[geo_df['satellite'] == 'TERRA', 'terra_weight'] = \
        geo_df.loc[geo_df['satellite'] == 'TERRA', 'confidence'] / 100.0
    geo_df.loc[geo_df['satellite'] == 'AQUA', 'aqua_weight'] = \
        geo_df.loc[geo_df['satellite'] == 'AQUA', 'confidence'] / 100.0
    geo_df.loc[geo_df['satellite'] == 'JP1_LATE', 'n20_weight'] = \
        geo_df.loc[geo_df['satellite'] == 'JP1_LATE', 'confidence'] / 100.0
    geo_df.loc[geo_df['satellite'] == 'NPP_LATE', 'npp_weight'] = \
        geo_df.loc[geo_df['satellite'] == 'NPP_LATE', 'confidence'] / 100.0

    # count number of Himawari observations
    geo_obs_count = int((end_date - start_date).seconds / 600)
    # normalize the weight for Himawari
    geo_df['geo_weight'] = geo_df['geo_weight'] / geo_obs_count

    # round to 8 decimals to save storage
    geo_df = geo_df.round(8)

    try:
        gdf = geopandas.GeoDataFrame(geo_df, geometry=geopandas.points_from_xy(geo_df.lon, geo_df.lat))
        log.debug('Created geopandas DataFrame')
    except Exception as e:
        log.exception(e)

    # transform to mercator epsg 3857
    gdf.crs = {'init': 'epsg:4326'}
    gdf_merc = gdf.to_crs({'init': 'epsg:3857'})

    gdf_merc.reset_index(inplace=True, drop=True)

    gdf_merc['x'] = gdf_merc['geometry'].x
    gdf_merc['y'] = gdf_merc['geometry'].y

    for key, value in sat_resolution_meter.items():
        gdf_merc.loc[gdf_merc['satellite'] == key, 'resolution_meter'] = value

    try:
        interim_file_path = os.path.join(args.out_file_path, 'interim')
        os.makedirs(interim_file_path, exist_ok=True)
    except Exception as e:
        log.exception(e)
        log.warning(interim_file_path + ' directory cannot be created!')

    try:
        processed_file_path = os.path.join(args.out_file_path, 'processed')
        os.makedirs(processed_file_path, exist_ok=True)
    except Exception as e:
        log.exception(e)
        log.warning(processed_file_path + ' directory cannot be created!')

    try:
        hotspot_json = os.path.join(interim_file_path, args.prefix_name + 'hotspot_'
                                    + end_date.strftime('%Y%m%d') + '.geojson')
        gdf_merc.to_file(hotspot_json, driver='GeoJSON')
        log.info(hotspot_json + ' is saved successfully.')
    except Exception as e:
        log.exception(e)
        log.warning(hotspot_json + ' export warning!')

    # create polygon
    for index, row in gdf_merc.iterrows():
        gdf_merc['geometry'].iloc[index] = get_poly_box(row['x'], row['y'], row['resolution_meter'])

    try:
        hotspot_polygon_json = os.path.join(interim_file_path, args.prefix_name +
                                            'hotspot_polygon_' + end_date.strftime('%Y%m%d') + '.geojson')
        gdf_merc.round(4)
        gdf_merc.to_file(hotspot_polygon_json, driver='GeoJSON')
        log.info(hotspot_polygon_json + ' is saved successfully.')
    except Exception as e:
        log.exception(e)
        log.warning(hotspot_polygon_json + ' export warning!')

    # for debugging
    # hotspot_polygon_json = os.path.join(interim_file_path, args.prefix_name +
    #                                     'hotspot_polygon_' + end_date.strftime('%Y%m%d') + '.geojson')
    # gdf_merc = geopandas.read_file(hotspot_polygon_json)

    try:
        log.debug('Processing grid sjoin...')
        df_grid_joined = geopandas.sjoin(df_grid, gdf_merc, op='intersects')
        grid_weight_total = df_grid_joined[['id', 'geo_weight', 'terra_weight',
                                            'aqua_weight', 'n20_weight', 'npp_weight']].groupby(['id']).sum()
        grid_geometry = df_grid_joined[['id', 'geometry']].groupby(['id']).first()
        processed_grid = pd.merge(grid_weight_total, grid_geometry, on='id')
        processed_grid_gpd = geopandas.GeoDataFrame(processed_grid)
        processed_grid_gpd.crs = {'init': 'epsg:3857'}
        log.debug('Processing grid completed.')
    except Exception as e:
        log.exception(e)
        log.error('Unable to process grid sjoin!')

    try:
        fusiun_ml_model = load(open(fusiun_ml_model_fpath, 'rb'))
        log.debug('Loaded trained model from ' + fusiun_ml_model_fpath)
        log.debug(f'Model pipeline: {fusiun_ml_model}')
        processed_grid_gpd['prob'] = fusiun_ml_model.predict_proba(processed_grid_gpd[fusiun_predict_features])[:, 1]
        log.info('Probabilities filled using FUSIUN Logistic Regression model.')
    except Exception as e:
        log.exception(e)

    try:
        hotspot_grid_json = os.path.join(processed_file_path, args.prefix_name + 'hotspot_grid_'
                                         + end_date.strftime('%Y%m%d') + '.geojson')
        processed_grid_gpd.to_file(hotspot_grid_json, driver='GeoJSON')
        log.info(processed_grid_gpd + ' is saved successfully.')
    except Exception as e:
        log.warning(hotspot_grid_json + ' export warning!')

    try:
        ann_file = os.path.join(processed_file_path, args.prefix_name + 'hotspot_grid_'
                                + end_date.strftime('%Y%m%d') + '.ann')
        save_fred_grid_meteor_ann(processed_grid_gpd, ann_file, low_prob_thres, med_prob_thres)
        log.info(ann_file + ' is saved successfully.')
    except Exception as e:
        log.exception(e)
        log.warning(ann_file + ' cannot be saved!')


if __name__ == "__main__":
    main()
