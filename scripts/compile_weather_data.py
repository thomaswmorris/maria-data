import glob
import logging
import pathlib
import re
import os
import h5py

import pytz
import warnings
import numpy as np
import netCDF4 as nc
import scipy as sp
import pandas as pd
import calendar

import time, os, stat

from datetime import datetime
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

def file_age_in_seconds(filepath):
    return time.time() - os.stat(filepath)[stat.ST_MTIME]

def get_utc_day_hour(t):
    dt = datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc)
    return dt.hour + dt.minute / 60 + dt.second / 3600

def get_utc_year_day(t):
    tt = datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc).timetuple()
    return (tt.tm_yday + get_utc_day_hour(t) / 24 - 1) 

def get_utc_year(t):
    return datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc).year

def grouper(iterable, tol=1):
    
    prev = None; group = []
    for item in iterable:
        if prev is None or item - prev <= tol: group.append(item)
        else: yield group; group = [item]
        prev = item
    if group: yield group




g = 9.80665

era5_base  = f'/users/tom/copernicus/era5'


all_files = glob.glob(f'{era5_base}/levels/hourly/*/*.nc')
for filepath in all_files:
    size = os.stat(filepath).st_size
    if size < 10e3: # greater than 10 kB
        print(f"File too small ({1e-3*size:.2f} KB): {filepath}")
    if (size > 100e3) and (size < 200e3): # greater than 10 kB
        print(f"Weird file ({1e-3*size:.2f} KB): {filepath}")
    
regions = pd.read_csv(f'/users/tom/copernicus/regions.csv', index_col=0)#.fillna('')
regions = regions.loc[regions.include_in_maria]

use_region = np.zeros(len(regions)).astype(bool)
regions['z']      = np.round(regions.altitude * g, 3)
for region, entry in regions.iterrows():
    
    regions.loc[region, 'n_hourly'] = len(glob.glob(f'{era5_base}/levels/hourly/{region}/*.nc'))
    # regions.loc[region, 'n_monthly'] = len(glob.glob(f'{era5_base}/monthly/levels/{region}/*.nc'))
   
n_days_min = 0

regions['use'] = (regions.n_hourly >= 365)# & (regions.n_monthly >= 12)


profile_translation = {
                    'z'     : 'geopotential',
                    'r'     : 'humidity',
                    't'     : 'temperature',
                    'u'     : 'wind_east',
                    'v'     : 'wind_north',
                    'ciwc'  : 'ice_water',
                    'clwc'  : 'liquid_water',
                    'cswc'  : 'snow_water',
                    'crwc'  : 'rain_water',
                    'o3'    : 'ozone',
                    'cc'    : 'cloud_cover',
                    'd'     : 'divergence',
                    'pv'    : 'potential_vorticity',
                    'vo'    : 'relative_vorticity',
                    'w'     : 'wind_vertical',
                    }

use_profile_variables = list(profile_translation.values())

derived_variables = ["wind_speed"]

quantiles = [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.]
quantile_data = {}

yd_bins = np.linspace(-1, 366, 25).astype(int).astype(float)
dh_bins = np.linspace(0, 24, 25)

n_yd = len(yd_bins) - 1
n_dh = len(dh_bins) - 1

year_day_edge_index = np.r_[-1,np.arange(n_yd),n_yd]
day_hour_edge_index = np.r_[-1,np.arange(n_dh),n_dh]

year_day_side = sp.interpolate.interp1d(np.arange(n_yd), .5*(yd_bins[1:] + yd_bins[:-1]), fill_value='extrapolate', kind='linear')(year_day_edge_index)
day_hour_side = sp.interpolate.interp1d(np.arange(n_dh), .5*(dh_bins[1:] + dh_bins[:-1]), fill_value='extrapolate', kind='linear')(day_hour_edge_index)

year_day_edge_index %= n_yd
day_hour_edge_index %= n_dh


data = {}

dtype = np.float32

for region, entry in regions.loc[regions.use].iterrows():

    print()

    region_file = f"/users/tom/copernicus/era5/levels/consolidated/{region}.h5"

    # if os.path.exists(region_file):
    #     if file_age_in_seconds(region_file) < 1e3:
    #         continue

    # if region not in ["thule"]: 
    #     continue
    
    
    data[region] = {}
    prefix = f'/users/tom'

    region_path = pathlib.Path(f"{era5_base}/levels/hourly/{region}")
    profile_filepaths = sorted(list(region_path.glob("*.nc")))
    profile_dates = [re.findall(r'/(.{10})\.nc', str(p_fp))[0] for p_fp in profile_filepaths]
    
    n_profile = 24 * len(profile_dates)
    # logger.info(f'{region} [n={len(profile_dates)}]')
       
    
    ######## PROFILE DATA ########

    init_profile_dataset = nc.Dataset(profile_filepaths[0])
    profile_variables = []

    data[region]['time'] = np.zeros(n_profile)
    for key in profile_translation.values():
        data[region][key] = np.zeros((n_profile, 37))


    sorted_dates = sorted(profile_dates)
    
    pbar = tqdm(range(len(sorted_dates)), desc=f"{region}")
    
    # pbar = enumerate(sorted(profile_dates))
    
    current_year = 0
    for idate in pbar:

        date = sorted_dates[idate]
        pbar.set_postfix(date=date)

        # year = datetime.fromisoformat(date).year
        # if year != current_year:
        #     pbar.set_description(f"{region} ({year})")
        #     current_year = year

        i0 = 24 * idate
        i1 = 24 * (idate + 1)
        
        timestamp = datetime.fromisoformat(date + 'T00:00:00').timestamp()     
        try:
            date_path = f"{era5_base}/levels/hourly/{region}/{date}.nc"
            profile_dataset = nc.Dataset(date_path)
        except:
            warnings.warn(f"Could not read file {date_path}.")
        for k, v in profile_translation.items():
            data[region][v][i0:i1] = profile_dataset[k][:,:,0,0].astype(dtype)

        if "valid_time" in profile_dataset.variables:
            data[region]['time'][i0:i1] = profile_dataset["valid_time"][:].data.astype(dtype) - 1800
        else:
            data[region]['time'][i0:i1] = 3600 * profile_dataset["time"][:].data.astype(dtype) + datetime(1900,1,1,0,0,0, tzinfo=pytz.utc).timestamp() - 1800
        
    data[region]['pressure_levels'] = (profile_dataset['level'] if "level" in profile_dataset.variables else profile_dataset.variables['pressure_level'])[:].data.astype(dtype)


    data[region]['lat'], data[region]['lon'] = profile_dataset['latitude'][:].mean(), profile_dataset['longitude'][:].mean()

    data[region]["day_hour"] = np.array(list(map(get_utc_day_hour, data[region]["time"])))
    data[region]["year_day"] = np.array(list(map(get_utc_year_day, data[region]["time"])))
    
    ######## COMPUTE SOME STUFF ########
    
    

    level_mask = np.arange(37) >= np.where(np.percentile(data[region]['geopotential'], q=99, axis=0) > entry.z)[0][0] - 1
    
    logger.info(f'keeping {level_mask.sum()} levels')
    
    for k in ['pressure_levels', *use_profile_variables]:
        data[region][k] = data[region][k][..., level_mask]
            
    isort = np.argsort(data[region]['time'])  
    data[region]['time'] = data[region]['time'][isort]

    for k, v in data[region].items():
        if k in use_profile_variables:  
            if np.isnan(np.atleast_1d(v)).any():
                raise ValueError(f'NaN in {region}:{k}')
            data[region][k] = data[region][k][isort]
    
    min_date = datetime.fromtimestamp(data[region]['time'].min() + 3600).astimezone(pytz.utc).ctime()#isoformat()[:19]
    max_date = datetime.fromtimestamp(data[region]['time'].max() + 3600).astimezone(pytz.utc).ctime()#isoformat()[:19]
    logger.info(f't_min = {min_date}')
    logger.info(f't_max = {max_date}')

    data[region]["wind_speed"] = np.sqrt(data[region]["wind_east"]**2 + data[region]["wind_north"]**2 + data[region]["wind_vertical"]**2)
    
    n_times = len(data[region]["time"])
    n_levels = len(data[region]["pressure_levels"])
    n_quantiles = len(quantiles)
    
    quantile_data[region] = {}

    pbar = tqdm([*use_profile_variables, *derived_variables], desc=f"computing quantiles for {region}")

    for v in pbar:

        pbar.set_postfix(variable=v)

        quantile_data[region][v] = np.zeros((n_yd, n_dh, n_quantiles, n_levels))

        yd_bin_index = np.digitize(data[region]["year_day"], yd_bins) - 1
        dh_bin_index = np.digitize(data[region]["day_hour"], dh_bins) - 1
        
        for uydbi in np.unique(yd_bin_index):
            yd_mask = yd_bin_index == uydbi
            for udhbi in np.unique(dh_bin_index):
                mask = yd_mask & (dh_bin_index == udhbi)
                
                quantile_data[region][v][uydbi, udhbi] = np.quantile(data[region][v][mask], q=quantiles, axis=0)



    climate = pd.DataFrame(columns=["median_low", "median_median", "median_high"])

    # year_day, day_hour, quantile, level
    for month_index in range(12):


        yd_bin_index = np.digitize(30 * month_index + 15, bins=yd_bins)

        month_name = calendar.month_name[month_index + 1]

        mmt = quantile_data[region]["temperature"][yd_bin_index, :, quantiles.index(0.5), 0]

        lmu = np.array([mmt.min(), mmt.mean(), mmt.max()])
        
        climate.loc[month_name] = (lmu - 273.15) * 1.8 + 32

    print()
    print(climate.round(1))
    print()

    h = data[region]['geopotential'] / g
    h_min = h[:, 0].max()
    h_max = h[:, -1].min()

    print(f"min_altitude = {int(entry.min_altitude)} m")
    print(f"max_altitude = {int(entry.max_altitude)} m")
    print(f"h_min = {int(h_min)} m")
    print(f"h_max = {int(h_max)} m")

    if h_min > entry.min_altitude + 500:
        print(f"WARNING: h_min = {int(h_min)} m, min_altitude = {int(entry.min_altitude)} m")


    with h5py.File(f"/users/tom/maria/data/atmosphere/weather/era5/{region}.h5", "w") as f:

        f.create_dataset("quantile_levels", data=quantiles, dtype=float)
        f.create_dataset("pressure_levels", data=data[region]["pressure_levels"], dtype=float)

        f.create_dataset("year_day_side", data=year_day_side)
        f.create_dataset("day_hour_side", data=day_hour_side)
        f.create_dataset("year_day_edge_index", data=year_day_edge_index)
        f.create_dataset("day_hour_edge_index", data=day_hour_edge_index)

        f.create_group("data")

        for v in [*use_profile_variables, *derived_variables]:

            f["data"].create_group(v)

            mean = quantile_data[region][v].mean(axis=2)[:, :, None, :]
            scale = quantile_data[region][v].std(axis=2)[:, :, None, :]
            scale = np.maximum(scale, 1e-12)

            normalized_quantiles = (quantile_data[region][v] - mean) / scale

            if np.isnan(normalized_quantiles).any():
                raise ValueError(f"variable '{v}' for region '{region}' has nan values.")

            f["data"][v].create_dataset("mean", data=mean, dtype="f")
            f["data"][v].create_dataset("scale", data=scale, dtype="f")
            f["data"][v].create_dataset("normalized_quantiles", data=normalized_quantiles, dtype="f", scaleoffset=4)


    # with h5py.File(f"/users/tom/maria/data/atmosphere/weather/era5/v2/{region}.h5", "w") as f:

    #     f.create_dataset("quantile_levels", data=quantiles, dtype=float)
    #     f.create_dataset("pressure_levels", data=data[region]["pressure_levels"], dtype=float)

    #     f.create_dataset("year_day_side", data=year_day_side)
    #     f.create_dataset("day_hour_side", data=day_hour_side)
    #     f.create_dataset("year_day_edge_index", data=year_day_edge_index)
    #     f.create_dataset("day_hour_edge_index", data=day_hour_edge_index)

    #     f.create_group("data")

    #     for v in [*use_profile_variables, *derived_variables]:

    #         min = quantile_data[region][v].min(axis=(0, 1))[None, None, :, :]
    #         max = quantile_data[region][v].max(axis=(0, 1))[None, None, :, :] + 1e-16

    #         normalized_quantiles = 255 * (quantile_data[region][v] - min) / (max - min)

    #         if np.isnan(normalized_quantiles).any():
    #             raise ValueError(f"variable '{v}' for region '{region}' has nan values.")

    #         f["data"][v].create_dataset("min", data=min, dtype="f")
    #         f["data"][v].create_dataset("max", data=max, dtype="f")
    #         f["data"][v].create_dataset("normalized_quantiles", data=normalized_quantiles, dtype=np.uint8)



    with h5py.File(region_file, "w") as f:
        
        for hk in ["time", "pressure_levels"]:
            f.create_dataset(hk, data=data[region][hk], dtype=float)
        
        f.create_group("data")
        
        for v in [*use_profile_variables, *derived_variables]:
            f["data"].create_dataset(v, data=data[region][v], dtype=float)


column_mapping = {
                'description': 'location',
                'country': 'country',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'altitude': 'altitude',
                'min_altitude': 'min_altitude',
                'max_altitude': 'max_altitude',
                'timezone':'timezone',
                }

maria_regions = regions.loc[regions.use].loc[:, list(column_mapping.keys())]

maria_regions.columns = list(column_mapping.values())

for col in ["latitude", "longitude", "altitude"]:
    maria_regions.loc[:, col] = maria_regions.loc[:, col].astype(float)

for region in data.keys():
    
    t_min, t_max = data[region]["time"].min() + 1800, data[region]["time"].max() + 1800
    
    date_min, date_max = [datetime.fromtimestamp(t).astimezone(pytz.utc).isoformat()[:10] for t in [t_min, t_max]]
    
    maria_regions.loc[region, 'training_start'] = date_min
    maria_regions.loc[region, 'training_end'] = date_max
    maria_regions.loc[region, 'training_years'] = np.round(len(data[region]["time"]) / (24 * 365.2422),1)
    
maria_regions = maria_regions.loc[~maria_regions.training_years.isna()]
maria_regions.to_csv(f"/users/tom/maria/src/maria/site/regions.csv")
maria_regions.to_csv(f"/users/tom/maria/data/regions.csv")

maria_regions