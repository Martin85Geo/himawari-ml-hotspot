#!/usr/bin/env python
"""
FRED - Multi-Satellite Fire Detection
Generates hotspot composite products based on hotspot detections from
multiple satellites

@author: Wong Songhan <Wong_Songhan@nea.gov.sg>
@date: 11 Dec 2019
@version: Beta 1.0

python generated_fred.py --start-date "2019-07-01 00:00" --end-date "2019-07-01 08:30"

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
from shapely.geometry import box



def save_fred_grid_meteor_ann(hs_df, output_file_path, w_minimum):
    """Gets coordinates from FRED fire grids GeoPandas DataFrame geometry
    and save into .ann (Polygon Annotation)

    Args:
        hs_df (obj): Hotspot GeoPandas DataFrame
        output_file_path (str): output file path
        w_minimum (float): threshold for minimum weight to be colored red

    """

    hs_df_wgs84 = hs_df.to_crs({'init': 'epsg:4326'})
    f = open(output_file_path, "w+")
    for index, row in hs_df_wgs84[hs_df_wgs84['adj_weight'] >= w_minimum].iterrows():
        x, y = row.geometry.exterior.coords.xy
        list_len = len(x)
        f.write('######################################\n')
        f.write('POLYLINE\n')
        f.write('COLOR 255   0   0\n')
        f.write('THICKNESS       2\n')
        f.write('LINESTYLE       0\n')
        f.write('STARTPOINTS\n')
        for i in range(0, list_len):
            f.write('       %f       %f\n' % (x[i], y[i]))
        f.write('ENDPOINTS\n')
    for index, row in hs_df_wgs84[hs_df_wgs84['adj_weight'] < w_minimum].iterrows():
        x, y = row.geometry.exterior.coords.xy
        list_len = len(x)
        f.write('######################################\n')
        f.write('POLYLINE\n')
        f.write('COLOR 255   255   0\n')
        f.write('THICKNESS       2\n')
        f.write('LINESTYLE       0\n')
        f.write('STARTPOINTS\n')
        for i in range(0, list_len):
            f.write('       %f       %f\n' % (x[i], y[i]))
        f.write('ENDPOINTS\n')

    f.write('######################################')
    f.close()


def get_poly_box(x, y, res_m):
    """Construct a Shapely Polygon square object based on the center coordinates and polygon square length

    Args:
        x (float): x coordinate in meter s
        y (float): y coordinate in meters
        res_m (float): polygon square length in meter

    Returns:
        poly (obj): Shapely Polygon object

    """
    delta_x = res_m * 0.5
    delta_y = res_m * 0.5
    poly = box(x - delta_x, y - delta_y, x + delta_x, y + delta_y)
    return poly


def main():
    logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)

    # Get the logger specified in the file
    f_handler = logging.FileHandler('generate_fred.log')
    f_handler.setLevel(logging.DEBUG)
    log = logging.getLogger(__name__)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    log.addHandler(f_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight-minimum', default=0.31, dest='w_minimum',
                        help='Minimum thresholds to be considered as fire grids')
    parser.add_argument('-s', '--start-date', dest='start_date_str', help='Start date')
    parser.add_argument('-e', '--end-date', dest='end_date_str', help='End date')
    parser.add_argument('--max-sunglint-angle', dest='max_sunglint_angle', default=0.0, type=float,
                        help='Sun glint angle is defined as angle between reflected solar path \
                             and satellite view angle. If hotspot has sun glint angle less than the max, \
                             the hotspot is rejected.')
    parser.add_argument('-a', '--alpha', dest='alpha', default=0.5, help='Alpha weight for polar orbiting satellites')
    parser.add_argument('--ahi-hotspot-folder', dest='ahi_folder', default=os.path.join('.', 'hotspots', 'ahi'),
                        help='AHI hotspot folder')
    parser.add_argument('--viirs-hotspot-folder', dest='viirs_folder', default=os.path.join('.', 'hotspots', 'viirs'),
                        help='VIIRS hotspot folder')
    parser.add_argument('--modis-hotspot-folder', dest='modis_folder', default=os.path.join('.', 'hotspots', 'modis'),
                        help='MODIS hotspot folder')
    parser.add_argument('-o', '--output-folder', dest="out_file_path", default=os.path.join('.', 'output'),
                        help="Specify output folder")
    parser.add_argument('-g', '--grid', dest="grid_shp", default=os.path.join('.', 'shapefile', '2km_grid_ASEAN.shp'),
                        help="Specify grid .shp file")
    parser.add_argument('-n', '--name', dest="prefix_name", default='NRT_2km_', help="Prefix for output file names")

    args = parser.parse_args()
    log.debug(args)
    alpha = args.alpha
    w_minimum = args.w_minimum
    max_sunglint_angle = args.max_sunglint_angle

    with open("config.json", "r") as read_file:
        json_config = json.load(read_file)

    clipping_box = json_config['parameters']['clipping_box']
    gamma = json_config['parameters']['gamma']
    sat_resolution_meter = json_config['sat_resolution_meter']
    shapefile_path = json_config['shapefile']['path']
    bounding_box = json_config['plotting']['bounding_box']
    dpi = json_config['plotting']['dpi']

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

    #remove hotspots outside of clipping area
    geo_hs.clip_hotspot(clipping_box)
    #reject hotspots due to sun glint
    geo_hs.reject_sunglint_hs('Himawari-8/9', max_sunglint_angle)

    # read in grid shapefile
    try:
        df_grid = geopandas.read_file(args.grid_shp)
        df_grid.crs = {'init': 'epsg:3857'}
        log.debug(args.grid_shp + ' loaded successfully!')
    except Exception as e:
        log.exception(e)
        log.error(args.grid_shp + ' cannot be loaded !')
        exit()

    geo_df = geo_hs.hs_df.copy()
    # count number of Himawari observations
    geo_obs_count = int((end_date - start_date).seconds / 600)
    start_date_str = start_date.strftime("%d/%m/%Y %H:%M:%S")
    end_date_str = end_date.strftime("%d/%m/%Y %H:%M:%S")
    # selects period of interest
    geo_df = geo_df[(geo_df['date'] >= start_date_str) & (geo_df['date'] <= end_date_str)]

    geo_df.loc[geo_df['confidence'] == 'NA', 'confidence'] = 0

    log.info('Assigning weights')
    num_polar_sat = len(geo_df.loc[geo_df['satellite'] != 'Himawari-8/9', 'satellite'].unique())
    geo_df['geo_weight'] = 0.0
    geo_df['polar_weight'] = 0.0

    geo_df.loc[geo_df['satellite'] == "Himawari-8/9", 'geo_weight'] = (1 / geo_obs_count) * (1 - alpha)

    if num_polar_sat >= 1:
        geo_df['polar_weight'] = (alpha / num_polar_sat) * (geo_df['confidence'] / 100)

    geo_df['weight'] = geo_df['geo_weight'] + geo_df['polar_weight']

    try:
        gdf = geopandas.GeoDataFrame(geo_df, geometry=geopandas.points_from_xy(geo_df.lon, geo_df.lat))
        log.debug('Created geopandas DataFrame')
    except Exception as e:
        log.exception(e)

    # transform to mercator epsg 3857
    gdf.crs = {'init': 'epsg:4326'}
    gdf_merc = gdf.to_crs({'init': 'epsg:3857'})
    gdf_merc.reset_index(inplace=True, drop=True)
    print (gdf_merc.head())

    gdf_merc['x'] = gdf_merc['geometry'].x
    gdf_merc['y'] = gdf_merc['geometry'].y

    for key, value in sat_resolution_meter.items():
        gdf_merc.loc[gdf_merc['satellite'] == key, 'resolution_meter'] = value

    try:
        hotspot_json = os.path.join(args.out_file_path, args.prefix_name + 'hotspot_'
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
        hotspot_polygon_json = os.path.join(args.out_file_path, args.prefix_name +
                                            'hotspot_polygon_' + end_date.strftime('%Y%m%d') + '.geojson')
        gdf_merc.to_file(hotspot_polygon_json, driver='GeoJSON')
        log.info(hotspot_polygon_json + ' is saved successfully.')
    except Exception as e:
        log.exception(e)
        log.warning(hotspot_polygon_json + ' export warning!')

    try:
        log.debug('Processing grid sjoin...')
        df_grid_joined = geopandas.sjoin(df_grid, gdf_merc, op='intersects')
        grid_weight_total = df_grid_joined[['id', 'weight', 'geo_weight', 'polar_weight']].groupby(['id']).sum()
        grid_geometry = df_grid_joined[['id', 'geometry']].groupby(['id']).first()

        processed_grid = pd.merge(grid_weight_total, grid_geometry, on='id')
        processed_grid_gpd = geopandas.GeoDataFrame(processed_grid)
        processed_grid_gpd.crs = {'init': 'epsg:3857'}
        processed_grid_gpd['weight'] = processed_grid_gpd['weight'].clip(0, 1)
        processed_grid_gpd['adj_weight'] = processed_grid_gpd['weight'] ** (1 / gamma)
        log.debug('Processing grid completed.')
    except Exception as e:
        log.exception(e)
        log.error('Unable to process grid sjoin!')
        exit()

    try:
        hotspot_grid_json = os.path.join(args.out_file_path, args.prefix_name + 'hotspot_grid_'
                                         + end_date.strftime('%Y%m%d') + '.geojson')
        processed_grid_gpd.to_file(hotspot_grid_json, driver='GeoJSON')
        log.info(processed_grid_gpd + ' is saved successfully.')
    except Exception as e:
        log.warning(hotspot_grid_json + ' export warning!')

    fire_grid_gpd = processed_grid_gpd[processed_grid_gpd['adj_weight'] >= w_minimum]

    try:
        ann_file = os.path.join(args.out_file_path, args.prefix_name + 'hotspot_grid_'
                                + end_date.strftime('%Y%m%d') + '.ann')
        save_fred_grid_meteor_ann(processed_grid_gpd, ann_file, w_minimum)
        log.info(ann_file + ' is saved successfully.')
    except Exception as e:
        log.exception(e)
        log.warning(ann_file + ' cannot be saved!')

    try:
        map_shape_feature = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree())
        log.debug(shapefile_path + ' loaded successfully!')
    except ImportError as e:
        log.exception(e)
        log.error('Unable to load ' + shapefile_path)
        exit()

    # begin plotting routine
    try:
        log.debug('begin plotting...')
        fig = plt.figure(figsize=(30, 30), dpi=dpi, frameon=False)
        ax = plt.axes([0, 0, 1, 1], projection=ccrs.Mercator())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        ax.set_extent(bounding_box, crs=ccrs.PlateCarree())
        #TODO: plotting configs to be set in .json
        ax.add_feature(map_shape_feature, linewidth=0.1, edgecolor='black', facecolor='None', zorder=1)
        ax.add_geometries(fire_grid_gpd.geometry, alpha=0.8, crs=ccrs.Mercator(), facecolor='none', edgecolor='red',
                          linewidth=0.1, zorder=2)
        log.debug('end plotting.')
        fire_grid_overlay_name = os.path.join(args.out_file_path, args.prefix_name + 'fire_grid_overlay_'
                                              + end_date.strftime('%Y%m%d') + '.png')
        plt.savefig(fire_grid_overlay_name, transparent=True, format='png', bbox_inches='tight', pad_inches=0)
        log.info(fire_grid_overlay_name + ' is saved.')
    except Exception as e:
        log.exception(e)
        log.error('Unable to save image!')


if __name__ == "__main__":
    main()
