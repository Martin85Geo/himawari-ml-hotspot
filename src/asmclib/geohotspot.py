import logging
import numpy as np
import pandas as pd
import geopandas
import glob
import os
from pyorbital import astronomy
from datetime import datetime, timedelta
from shapely import geometry

logging.config.fileConfig(fname=os.path.join('config', 'log.config'), disable_existing_loggers=False)

# Get the logger specified in the file
f_handler = logging.FileHandler(os.path.join('logs', __name__ + '.log'))
f_handler.setLevel(logging.DEBUG)
log = logging.getLogger(__name__)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
log.addHandler(f_handler)


def get_MOD14_date_sat(filename):
    """Get the date and name of satellite based on the MOD14 file naming convention
    Args:
        filename (str): Filename

    Returns:
        sat_date_obj (obj), satellite (str): Tuple
    """

    satname_dict = {'MOD14': 'TERRA', 'MYD14': 'AQUA'}
    satname_str = filename[0:5]
    date_str = filename[6:17]

    sat_date_obj = datetime.strptime(date_str, "%y%j%H%M%S")
    satellite = satname_dict[satname_str]

    return (sat_date_obj, satellite)


def day_or_night(HH):
    """Check if the hour is day or nighttime

    Args:
        HH: Hour of the day 

    Returns:
        'day' or 'night' str 
    """
    if HH >= 0 and HH <= 11:
        return 'day'
    elif HH > 11:
        return 'night'


def datetime_from_utc_to_local(year, month, day, hour, minute):
    """
    Converts datetime from utc to local time (+8 hour for Singapore)

    Args:
        year (int):
        month (int):
        day (int):
        hour (int):
        minute (int):

    Returns:
        datetime (obj)
    """

    offset = timedelta(hours=8)
    return datetime(year, month, day, hour, minute) + offset


def compute_sun_glint_angle(sat_angle):
    """Compute the sun glint angle. Sun glint angle is defined as the angle difference between reflected solar beam and
    the satellite view/zenith angle, from the satellite reference frame. If the angle is small, it means that the
    reflected solar beam is seen by the satellite as sun glint.

    The eqn of sun glint is
        cos (theta_g) = cos (theta_v) * cos (theta_s) - sin (theta_v) * sin (theta_s) * cos (phi), where
        theta_g = sun glint angle, theta_v = satellite view/zenith angle, theta_s = solar zenith angle,
        phi = relative azimuth angle (difference between solar and satellite azimuth angle)

    For more info, refer to "An Enhanced Contextual Fire Detection Algorithm for MODIS" by Louis Giglio et al 2003
    https://doi.org/10.1016/S0034-4257(03)00184-6

    Args:
        sat_angle (dict): Dict of satellite angles: {
                        'viewzenang' (float): satellite view/zenith angle,
                        'solarzenang' (float): solar zenith angle,
                        'relazang' (float): relative azimuth angle ,
                         }

    Returns:
        sun_glint_angle (float): Angle of sun glint in degrees

    """
    theta_v = np.radians(sat_angle['viewzenang'])
    theta_s = np.radians(sat_angle['solarzenang'])
    phi = np.radians(sat_angle['relazang'])
    cos_sunglint = np.cos(theta_v) * np.cos(theta_s) - (np.sin(theta_v) * np.sin(theta_s) * np.cos(phi))
    sun_glint_rad = np.arccos(cos_sunglint)
    sun_glint_angle = np.degrees(sun_glint_rad)

    return sun_glint_angle


class GeoHotspot:
    """GeoHotspot is a class with readers to read hotspot data from different satellites and store the
    hotspot attributes in a Pandas DataFrame

    Attributes:
        MAX_DIST (CONST): Maximum tolerance (in deg) for points outside polygon, this is used only for handling Lexical
                          hotspot data as the water masking is imperfect
        hs_df (obj): Pandas DataFrame to store hotspot attributes
        gdf (obj): GeoPandas DataFrame to store hotspot data after 'sjoin' operations with region/subregion
    """

    def __init__(self):
        # 5 km tolerance for points outside the polygon
        self.MAX_DIST = 0.05
        self.hs_df = pd.DataFrame()
        self.gdf = geopandas.GeoDataFrame()

    def parse_lexical_txt(self, file_path, convert_local_dt=False):
        """
        Parse the Lexical hotspot text file
    
        Returns:
            Pandas DataFrame with coordinates, satellite of hotspots, timestamp 
        """
        i = -1

        hs_lexical_df = pd.DataFrame()

        # load the txt file
        # print ("searching for " + file_path)
        log.debug("searching for " + file_path)
        for file in glob.glob(file_path):
            try:
                f = open(file, "r")
                # print (file + ' text succesfully opened.')

                # parse through content of txt file
            except Exception as e:
                log.exception(e)
                log.debug(file_path + " not found.")

            for line in f:
                hs_series = pd.Series()

                # trigger for parsing of hotspot coordinates
                if line.startswith("="):
                    i = i * -1
                    continue

                if line.startswith("Satellite"):
                    str_satellite = line.split("Satellite:")[1]

                if line.startswith("Date & Time"):
                    str_date = line.split("Date & Time:")[1]
                    int_year = int(str_date.split()[0][:4])
                    int_mm = int(str_date.split()[0][5:7])
                    int_dd = int(str_date.split()[0][8:10])
                    int_HH = int(str_date.split()[1][:2])
                    int_MM = int(str_date.split()[1][3:5])

                if i > 0:
                    hs_series['lon'] = float(line.strip().split()[1])
                    hs_series['lat'] = float(line.strip().split()[2])
                    hs_series['satellite'] = str_satellite.split()[0]

                    if convert_local_dt:
                        hs_series['date'] = datetime_from_utc_to_local(int_year, \
                                                                       int_mm, int_dd, int_HH, int_MM) \
                            .strftime("%d/%m/%Y %H:%M:%S")
                    else:
                        hs_series['date'] = datetime(int_year, \
                                                     int_mm, int_dd, int_HH, int_MM) \
                            .strftime("%d/%m/%Y %H:%M:%S")

                    hs_series['confidence'] = "NA"
                    hs_series['FRP'] = "NA"
                    hs_series['daynight'] = day_or_night(int_HH)
                    hs_lexical_df = hs_lexical_df.append(hs_series, \
                                                         ignore_index=True)

        self.hs_df = pd.concat([hs_lexical_df, self.hs_df])

    def parse_jaxa_hotspot_txt(self, file_path):
        """
        Parse the JAXA Himawari-8/9 AHI hotspot text and insert into the Pandas DataFrame
        with coordinates, fire radiative power, detection confidence, timestamp and satellite of hotspots.

        Args:
              file_path (str): File path to the JAXA Himawari-8/9 hotspot .csv
        """
        hs_ahi_df = pd.DataFrame()
        cols_to_use_list = [0, 2, 7, 8, 9, 10]
        date_col_list = [0]

        # load the txt file
        log.debug("searching for " + file_path)

        for file in glob.glob(file_path):
            try:
                f = open(file, "r")
                filename = os.path.basename(file)
                log.debug(file + " text succesfully opened.")
            except Exception as e:
                log.exception(e)
                log.debug(file_path + " not found.")

            temp_hs_ahi_df = pd.read_csv(file, sep=",", skiprows=[0], \
                                         header=None, usecols=cols_to_use_list, \
                                         names=['date', 'satellite', 'lon', 'lat', 'viewzenang', 'viewazang'], \
                                         parse_dates=date_col_list)

            temp_hs_ahi_df['confidence'] = 'NA'
            temp_hs_ahi_df['FRP'] = 'NA'
            temp_hs_ahi_df['satellite'] = 'Himawari-8/9'

            if len(temp_hs_ahi_df) > 0:
                try:
                    log.debug("Compute solar azimuth and zenith angle, and relative zenith angle.")
                    temp_hs_ahi_df['solarazang'] = temp_hs_ahi_df.apply( \
                        lambda x: np.degrees(astronomy.get_alt_az(x['date'], x['lon'], x['lat'])[1]), axis=1)
                    temp_hs_ahi_df['solarzenang'] = temp_hs_ahi_df.apply( \
                        lambda x: astronomy.sun_zenith_angle(x['date'], x['lon'], x['lat']), axis=1)
                    temp_hs_ahi_df['relazang'] = temp_hs_ahi_df['solarazang'] - temp_hs_ahi_df['viewazang']
                    temp_hs_ahi_df['sunglint_angle'] = temp_hs_ahi_df.apply(compute_sun_glint_angle, axis=1)
                except Exception as e:
                    log.warning("Unable to compute solar azimuth and zenith angle, and relative zenith angle.")
                    log.exception(e)

                try:
                    temp_hs_ahi_df.loc[(temp_hs_ahi_df['date'].dt.hour >= 0) \
                                       & (temp_hs_ahi_df['date'].dt.hour <= 11), 'daynight'] = 'day'
                    temp_hs_ahi_df.loc[temp_hs_ahi_df['date'].dt.hour > 11, \
                                       'daynight'] = 'night'
                    temp_hs_ahi_df['date'] = temp_hs_ahi_df['date'].dt.strftime( \
                        "%d/%m/%Y %H:%M:%S")
                except Exception as e:
                    log.warning("Warning encountered parsing date from %s." % file)
                    date_from_file = datetime.strptime(filename[4:17], "%Y%m%d_%H%M")
                    log.warning("Parse date from filename instead. %s" % date_from_file)
                    temp_hs_ahi_df['date'] = date_from_file.strftime("%d/%m/%Y %H:%M:%S")

            hs_ahi_df = pd.concat([hs_ahi_df, temp_hs_ahi_df])

        if len(hs_ahi_df) > 0:
            hs_ahi_df = hs_ahi_df.reset_index(drop=True)

        self.hs_df = pd.concat([hs_ahi_df, self.hs_df])

    def parse_modis_mod14_txt(self, file_path, convert_local_dt=False):
        """
        Parse the MODIS MOD14 hotspot txt and insert into a Pandas DataFrame with coordinates, fire radiative power,
        detection confidence, timestamp and satellite of hotspots

        Args:
            file_path (str): File path
            convert_local_dt (bool): Whether to convert datetime to local SGT

        """

        hs_modis_df = pd.DataFrame()

        # load the txt file
        log.debug("searching for " + file_path)

        for file in glob.glob(file_path):
            try:
                f = open(file, "r")
                log.debug(file + " text succesfully opened.")
            except:
                log.debug(file_path + " not found.")

                # parse through content of txt file
            filename = os.path.basename(file)
            sat_date_obj, sat_name = get_MOD14_date_sat(filename)

            if convert_local_dt:
                sat_date_obj = sat_date_obj + timedelta(hours=8)

            temp_hs_modis_df = pd.read_csv(file, header=None, usecols=[0, 1, 5, 6], \
                                           names=['lat', 'lon', 'confidence', 'FRP'])

            temp_hs_modis_df['satellite'] = sat_name
            temp_hs_modis_df['date'] = sat_date_obj.strftime("%d/%m/%Y %H:%M:%S")
            temp_hs_modis_df['daynight'] = day_or_night(sat_date_obj.hour)

            hs_modis_df = pd.concat([hs_modis_df, temp_hs_modis_df])

        if len(hs_modis_df) > 0:
            hs_modis_df = hs_modis_df.reset_index(drop=True)

        hs_modis_df.drop_duplicates(['lat', 'lon', 'confidence', 'FRP', 'daynight'], keep='first', inplace=True)

        self.hs_df = pd.concat([hs_modis_df, self.hs_df])

    def parse_viirs_afedr_txt(self, file_path, sat_name, convert_local_dt=False):
        """Parse the CSPP Active Fires EDR hotspot text and append to
        a Pandas DataFrame with coordinates, fire radiative power, detection confidence,
        timestamp and satellite of hotspots

        Args:
            file_path (str): File path for input .txt
            sat_name (str): Name of satellite (e.g NOAA-20, NPP)
            convert_local_dt (bool): Whether to convert datetime to local SGT
        """
        hs_viirs_df = pd.DataFrame()

        # load the txt file
        log.debug("searching for " + file_path)

        for file in glob.glob(file_path):
            try:
                f = open(file, "r")
                log.debug(file + " text succesfully opened.")
            except:
                log.debug(file_path + " not found.")

            # parse through content of txt file
            str_date = os.path.basename(file)[0:12]
            int_year = int(str_date[:4])
            int_mm = int(str_date[4:6])
            int_dd = int(str_date[6:8])
            int_HH = int(str_date[8:10])
            int_MM = int(str_date[10:12])
            temp_hs_viirs_df = pd.read_csv(file, delimiter=",", header=None, comment='#',
                                           names=['lat', 'lon', 'M13BT', 'ASpx', \
                                                  'ATpx', 'confidence', 'FRP', 'satellite', 'date'])
            temp_hs_viirs_df.drop(['M13BT', 'ASpx', 'ATpx'], 1, inplace=True)

            if convert_local_dt:
                temp_hs_viirs_df['date'] = datetime_from_utc_to_local(int_year, int_mm,
                                                                      int_dd, int_HH, int_MM) \
                    .strftime("%d/%m/%Y %H:%M:%S")
            else:
                temp_hs_viirs_df['date'] = datetime(int_year, int_mm,
                                                    int_dd, int_HH, int_MM) \
                    .strftime("%d/%m/%Y %H:%M:%S")

            temp_hs_viirs_df['daynight'] = day_or_night(int_HH)
            temp_hs_viirs_df['satellite'] = sat_name

            hs_viirs_df = pd.concat([hs_viirs_df, temp_hs_viirs_df])

        if len(hs_viirs_df) > 0:
            hs_viirs_df = hs_viirs_df.reset_index(drop=True)

        hs_viirs_df.drop_duplicates(['lat', 'lon', 'confidence', 'FRP', 'daynight'], keep='first', inplace=True)

        self.hs_df = pd.concat([hs_viirs_df, self.hs_df])

    def get_hs_from_csv_db(self, file_path, sat_name, date_start, date_end, convert_to_utc=True):
        """
        Reader of .csv db exported by Microsoft Access
        List of headers:
        'ID', 'satellite', 'datestamp', 'lat', 'lon', 'FRP', 'confidence', 'daynight', 'region', 'subregion'

        Args:
            file_path (str): Path to .csv
            sat_name (str): Name of satellite
            date_start (obj): Start of date obj (UTC)
            date_end (obj): End of date obj (UTC)
            convert_to_utc (bool): Original data is in local time, convert to UTC
        """

        hs_df = pd.DataFrame()
        cols_to_use_list = [1, 2, 3, 4, 5, 6, 7]
        date_col_list = [1]

        # load the txt file
        log.debug("searching for " + file_path)

        for file in glob.glob(file_path):
            try:
                f = open(file, "r")
                log.debug(file + " text succesfully opened.")
            except:
                log.debug(file_path + " not found.")

            # parse through content of txt file
            temp_hs_df = pd.read_csv(file, sep=",", header=0, usecols=cols_to_use_list, \
                                     names=['satellite', 'date', 'lat', 'lon', 'FRP', 'confidence', 'daynight'], \
                                     parse_dates=date_col_list, infer_datetime_format=True)

            if convert_to_utc:
                temp_hs_df['date'] = temp_hs_df['date'] - timedelta(hours=8)

            temp_hs_df = temp_hs_df[(temp_hs_df['date'] >= date_start) & (temp_hs_df['date'] <= date_end)]
            temp_hs_df['date'] = temp_hs_df['date'].dt.strftime("%d/%m/%Y %H:%M:%S")
            temp_hs_df = temp_hs_df[temp_hs_df['satellite'] == sat_name]

            hs_df = pd.concat([hs_df, temp_hs_df])

        if len(hs_df) > 0:
            hs_df = hs_df.reset_index(drop=True)

        self.hs_df = pd.concat([hs_df, self.hs_df])

    def count_hs_shp(self, asean_shp_path):
        """Compute number of hotspots for each region/subregion in the shapefile provided. Assign the results to the
        GeoPandas DataFrame `gdf` attr.

        Args:
            asean_shp_path (str): File path to the shapefile

        """
        try:
            # for geopandas
            asean_shp = geopandas.read_file(asean_shp_path)
            asean_shp.rename(columns={'NAME_0': 'region', 'NAME_1': 'subregion'}, \
                             inplace=True)
            log.debug(asean_shp_path + " shapefile succesfully loaded.")
        except Exception as e:
            log.exception(e)
            log.debug(asean_shp_path + " shapefile not loaded.")

        hs_df = self.hs_df

        if len(hs_df) > 0:
            log.info("processing count for %d hotspots" % (len(hs_df)))
            hs_df.drop_duplicates(keep='first', inplace=True)
            hs_df.reset_index(drop=True, inplace=True)

            hs_df['Coordinates'] = list(zip(hs_df.lon, hs_df.lat))
            hs_df['Coordinates'] = hs_df['Coordinates'].apply(geometry.Point)

            log.debug("executing spatial join")

            try:
                gdf = geopandas.GeoDataFrame(hs_df, geometry='Coordinates')
                gdf.crs = {'init': 'epsg:4326'}
                gdf_joined = geopandas.sjoin(gdf, asean_shp, op='within', how='left')
                gdf_joined.drop(columns={'index_right'}, inplace=True)
                gdf_joined.fillna('NA', inplace=True)
            except Exception as e:
                log.error(e)
                log.error("Error encountered performing sjoin operations.")

            # special handling of coordinate falls within 5 km radius of nearest feature
            # for lexical output data
            for index, row in gdf_joined[((gdf_joined.satellite == 'NOAA19') | \
                                          (gdf_joined.satellite == 'Terra') | \
                                          (gdf_joined.satellite == 'Aqua')) & \
                                         (gdf_joined.region == 'NA')].iterrows():
                hs_pt = geometry.Point(row['lon'], row['lat'])
                dist_arr = []

                # check through each shapefile polygon
                for i, asean_shp_feature in asean_shp.iterrows():
                    region_poly = geometry.shape(asean_shp_feature['geometry'])
                    dist_arr.append(hs_pt.distance(region_poly))

                min_dist = min(dist_arr)

                index_min = dist_arr.index(min_dist)

                # check for maximum distance, reject if point is too far
                if min_dist < self.MAX_DIST:
                    gdf_joined.loc[index, 'region'] = asean_shp.loc[index_min, 'region']
                    gdf_joined.loc[index, 'subregion'] = asean_shp.loc[index_min, 'subregion']

                del hs_pt, dist_arr, min_dist, index_min

            gdf_joined['count'] = 1
            gdf_joined.drop(columns={'Coordinates'}, inplace=True)
            gdf_joined.drop_duplicates(keep='first', inplace=True)
        else:
            gdf_joined = []

        self.gdf = gdf_joined

    def reject_sunglint_hs(self, sat_name, max_sunglint_angle):
        """Remove hotspot from `hs_df` attr where `sunglint_angle` is smaller than `max_sunglint_angle`

        Args:
            sat_name (str): Name of satellite
            max_sunglint_angle (float): Maximum sun glint angle
        """
        try:
            self.hs_df.reset_index(inplace=True, drop=True)
            flag_reject = (self.hs_df['sunglint_angle'] <= max_sunglint_angle) & (self.hs_df['satellite'] == sat_name)
            ratio_reject = flag_reject.sum() / len(self.hs_df)
            log.info("Maximum sun glint angle for rejection: %f" %max_sunglint_angle)
            log.info("Rejecting {:.2%} of hotspots due to sun glint.".format(ratio_reject))
            self.hs_df.drop(self.hs_df.loc[flag_reject].index, inplace=True)
        except Exception as e:
            log.exception(e)
            log.warning("Error encountered when rejecting hotspots due to sun glint!")

    def clip_hotspot(self, clipping_box):
        #TODO: fix the bug!
        """Remove hotspot from `hs_df` where `lat` and `lon` are outside of the clipping box

        Args:
            clipping_box (arr): [lon_min, lon_max, lat_min, lat_max]
        """

        try:
            self.hs_df.reset_index(inplace=True, drop=True)
            flag_keep = (self.hs_df['lat'] <= clipping_box[3]) & (self.hs_df['lat'] >= clipping_box[2]) & \
                          (self.hs_df['lon'] <= clipping_box[1]) & (self.hs_df['lon'] >= clipping_box[0])
            #use bitwise operator ~ to invert boolean series for flag_keep
            log.debug("Remove hotspots outside of clipping box %s" %clipping_box)
            self.hs_df.drop(self.hs_df.loc[~flag_keep].index, inplace=True)
        except Exception as e:
            log.exception(e)
            log.warning("Error encountered when clipping hotspots to clipping_box.")
