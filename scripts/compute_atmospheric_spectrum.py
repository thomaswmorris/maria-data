import time
import h5py
import os
import pathlib
import logging
import subprocess
import re, os, stat
import argparse
import numpy as np
import scipy as sp
import pandas as pd
from maria.weather import relative_to_absolute_humidity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("am")

parser = argparse.ArgumentParser()
parser.add_argument("--region", type=str, help="maria region")
parser.add_argument("--tag", type=str)
args = parser.parse_args()

region = args.region
tag = args.tag


spec_ranges = [(1e6, 1e7, 1e-1), # 1 MHz to 10 MHz
               (1e7, 1e8, 1e-1), # 10 MHz to 100 MHz
               (1e8, 1e9, 1e-1), # 100 MHz to 1 GHz
               (1e9, 1e10, 1e-2), # 1 GHz to 10 GHz
               (1e10, 1e11, 1e-2), # 10 GHz to 100 GHz
               (1e11, 1e12, 1e-3), # 100 GHz to 1 THz
               (1e12, 15e12, 1e-2)] # 1 THz to 15 THz

# spec_ranges = [
#     (1e6, 2e6, 1e-1),
#     (2e6, 5e6, 1e-1),
#     (5e6, 1e7, 1e-1),
#     (1e7, 2e7, 1e-1),
#     (2e7, 5e7, 1e-1),
#     (5e7, 1e8, 1e-1),
#     (1e8, 2e8, 1e-1),
#     (2e8, 5e8, 1e-1),
#     (5e8, 1e9, 1e-1),
#     (1e9, 2e9, 1e-2),
#     (2e9, 5e9, 1e-2),
#     (5e9, 1e10, 1e-2),
#     (1e10, 2e10, 1e-3),
#     (2e10, 5e10, 1e-3),
#     (5e10, 1e11, 1e-3),
#     (1e11, 2e11, 1e-3),
#     (2e11, 5e11, 1e-3),
#     (5e11, 1e12, 1e-2),
#     (1e12, 2e12, 1e-1),
#     (2e12, 5e12, 1e-1),
#     (5e12, 1e13, 1e-1),
#     (1e13, 1.5e13, 1e-1),
# ]

decades = []
nu = np.empty(0)
cum_n = 0
for i, (f_min, f_max, rel_step) in enumerate(spec_ranges):

    f_step = rel_step * f_min
    f_max = f_max if i + 1 == len(spec_ranges) else f_max - f_step
    
    decade_nu = np.arange(f_min, f_max + 1e0, step=f_step)
    decades.append({"f_min": int(f_min), 
                    "f_max": int(f_max), 
                    "f_step": int(f_step), 
                    "n": len(decade_nu),
                    "start_index": cum_n})
    cum_n += len(decade_nu)
    nu = np.r_[nu, decade_nu]

# for running on della
MARIA_PATH = "/users/tom/maria"
AM_PATH = "/users/tom/am-14.0/src/am"

REGIONS = pd.read_csv("/users/tom/maria/data/regions.csv", index_col=0)
region_entry = REGIONS.loc[region]

write_dir = f"{MARIA_PATH}/raw_spectra/{tag}"
assert os.path.isdir(write_dir)
write_path = f"{write_dir}/{region}.h5"

if os.path.exists(write_path):
    if time.time() - os.stat(write_path)[stat.ST_MTIME] < 7 * 86400: # one week:
        print(f"skipping {write_path}")
        quit()

with h5py.File(write_path, "w") as f:
    ...

h_master = np.arange(0, 45000 + 1, 250)

profiles = {}
profiles["temperature"] = {}
profiles["pressure"] = {}
profiles["ozone"] = {}
profiles["absolute_humidity"] = {}

quantiles = {}
spectra_data = {}

with h5py.File(f"{MARIA_PATH}/data/atmosphere/weather/era5/{region}.h5", "r") as f:
    fields = list(f["data"].keys())

    pressure_profile = 1e2 * f["pressure_levels"][:] # in Pa
    quantile_levels = f["quantile_levels"][:]

    for attr in fields:
        quantiles[attr] = f["data"][attr]["normalized_quantiles"][:] * f["data"][attr]["scale"][:] + f["data"][attr]["mean"][:]


with h5py.File(f"/users/tom/era5/consolidated/{region}.h5", "r") as f:

    d = {}
    
    h_data = f["data"]["geopotential"][:].mean(axis=0) / 9.8
    d["temperature"] = f["data"]["temperature"][:].mean(axis=0)
    d["ozone"] = f["data"]["ozone"][:].mean(axis=0)
    d["pressure"] = np.log(1e2 * f["pressure_levels"][:])
    
    relative_humidity = f["data"]["humidity"][:].mean(axis=0)
    d["absolute_humidity"] = np.log(relative_to_absolute_humidity(d["temperature"], relative_humidity))

mask = h_data < 1e4 #h_data.min() + 5e3

for k, data in d.items():

    linear = lambda x, a, b: a * x + b
    pars, cpars = sp.optimize.curve_fit(linear, h_data[mask], data[mask])
    
    h_interp = [0, *h_data]
    d_interp = [linear(0, *pars), *data]
    
    profiles[k][region] = np.interp(h_master, h_interp, d_interp)

for key in ["pressure", "absolute_humidity"]:
    profiles[key][region] = np.exp(profiles[key].pop(region))

    
# h_from_canonical = np.linspace(region_entry.altitude, h_data.max(), 1024)
# w_from_canonical = np.interp(h_from_canonical, h_master, profiles["absolute_humidity"][region])

# typical_pwv = np.trapezoid(w_from_canonical, x=h_from_canonical)

if region_entry.max_altitude - region_entry.min_altitude <= 1e3:
    altitude_samples = [region_entry.min_altitude, region_entry.max_altitude]
else:
    altitude_samples = [region_entry.min_altitude, region_entry.altitude, region_entry.max_altitude]
    
zenith_pwv_samples = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
base_temperature_samples = np.percentile(quantiles["temperature"][..., 0], q=[0, 100])
elevation_samples = np.linspace(10, 90, 9)

# pwv_scales = zenith_pwv_samples / typical_pwv

# base_pressure_samples = np.percentile(quantiles["temperature"][..., 0], q=[0, 50, 100])
# elevation_samples = np.linspace(5, 90, 18)
# elevation_samples = np.linspace(10, 90, 9)

TRJ = np.zeros((
            len(altitude_samples),
            len(base_temperature_samples),
            len(zenith_pwv_samples), 
            len(elevation_samples),
            len(nu)))
TAU = np.zeros(TRJ.shape)
L = np.zeros(TRJ.shape)

total_spectra = np.prod(TRJ.shape[:-1]) * len(decades)
i_spectrum = 0


region_start = time.monotonic()

for i_alt, alt in enumerate(altitude_samples):

    layer_boundaries = [alt]

    while max(layer_boundaries) < h_master.max():

        layer_res = np.interp(layer_boundaries[-1], 
                              [alt, h_master.max()], 
                              [100, 2000])

        layer_boundaries.append(layer_boundaries[-1] + layer_res)

    layer_boundaries = np.array(layer_boundaries)
    layer_middles = (layer_boundaries[1:] + layer_boundaries[:-1]) / 2
    layer_abs_hum = np.interp(layer_middles, h_master, profiles["absolute_humidity"][region])
    typical_pwv_per_layer = np.diff(layer_boundaries) * layer_abs_hum

    layer_tbase = np.interp(layer_boundaries[1:], h_master, profiles["temperature"][region])
    layer_pbase = np.interp(layer_boundaries[1:], h_master, profiles["pressure"][region])
    layer_ozone = (28.96 / 48) * np.interp(layer_boundaries[1:], h_master, profiles["ozone"][region])
    
    for i_bt, bt in enumerate(base_temperature_samples):      
        for i_pwv, pwv in enumerate(zenith_pwv_samples):
            for i_el, el in enumerate(elevation_samples):

                layer_tbase = layer_tbase + bt - layer_tbase[0]
                layer_pwvs = typical_pwv_per_layer * (pwv / typical_pwv_per_layer.sum())

                for i_decade, decade in enumerate(decades):

                    start_index, n = decade["start_index"], decade["n"]

                    config_header = f"""
    f {decade['f_min']} Hz {decade['f_max']} Hz {decade['f_step']} Hz
    output f Hz Trj K tau neper L m
    tol 0
    za {90 - el} deg
    T0 0 K
    """
                    layer_configs = []

                    for i_layer, hbase in enumerate(layer_boundaries[1:]):

                        layer_configs.append(f"""
    layer
    Pbase {layer_pbase[i_layer]:.01f} Pa
    Tbase {layer_tbase[i_layer]:.01f} K
    column h2o {1e3 * layer_pwvs[i_layer]:.03f} um_pwv
    column o3 vmr {layer_ozone[i_layer]:.01e}
    column dry_air vmr""")

                    config_text = config_header + "\n".join(layer_configs[::-1])
        
                    pathlib.Path(f"/tmp/am/{region}").mkdir(parents=True, exist_ok=True)
                    config_path = f"/tmp/am/{region}/config.amc"
                    with open(config_path, "w") as f:
                        f.write(config_text)

                    start_time = time.monotonic()
                    fails = 0

                    while fails < 7:
                        try:
                            proc = subprocess.run([AM_PATH, config_path], capture_output=True, text=True)
                            spec = pd.DataFrame(np.array(proc.stdout.split()).reshape(-1,4).astype(float), columns=["nu", "trj", "tau", "L"])
                            # zpwvstr, lospwvstr = re.findall(r"# *\((.+) um_pwv\) *\((.+) um_pwv\) *", proc.stderr)[0]
                            # zpwv, lospwv = float(zpwvstr), float(lospwvstr)

                            TAU[i_alt, i_bt, i_pwv, i_el, start_index:start_index+n] = spec.tau
                            TRJ[i_alt, i_bt, i_pwv, i_el, start_index:start_index+n] = spec.trj
                            L[i_alt, i_bt, i_pwv, i_el, start_index:start_index+n] = spec.L

                            break
                        except Exception as e:
                            print(e)
                            print()
                            print(proc.stderr)
                            time.sleep(1e0)
                            fails += 1
                    else:
                        raise ValueError("Too many fails.")

                    i_spectrum += 1

                    mtpl = (time.monotonic() - region_start) / i_spectrum

                    expected_finish_time = region_start + total_spectra * mtpl

                    logger_data = {"region": region,
                                   "alt": f"{alt} m",
                                   "base_temp": f"{bt:.1f} K",
                                   "pwv": f"{pwv:.02f} mm",
                                   "elev": f"{el} deg",
                                   "nu": f"{decade['f_min']:.01e}Hz-{decade['f_max']:.01e}Hz",
                                   "duration": f"{time.monotonic() - start_time:.02f}",
                                   "etr": f"{expected_finish_time - time.monotonic():.1f} s"}
                    
                    logger.info(" | ".join([f"{k}={v}" for k, v in logger_data.items()]) + f" | {i_spectrum} / {total_spectra}")



spectra_data["side_nu_Hz"] = np.array(nu)
spectra_data["side_elevation_deg"] = np.array(elevation_samples)
spectra_data["side_zenith_pwv_mm"] = np.array(zenith_pwv_samples)
spectra_data["side_base_temperature_K"] = np.array(base_temperature_samples)
spectra_data["side_altitude_m"] = np.array(altitude_samples)
spectra_data["rayleigh_jeans_temperature_K"] = TRJ
spectra_data["opacity_nepers"] = TAU
spectra_data["excess_path_m"] = L


with h5py.File(write_path, "w") as f:

    for key in ["side_nu_Hz", "side_zenith_pwv_mm", "side_base_temperature_K", "side_elevation_deg", "side_altitude_m"]:

        f.create_dataset(key, data=spectra_data[key].astype(float), dtype="f")

    output_config = {
        "rayleigh_jeans_temperature_K": {"kwargs":{"scaleoffset": 4}},
        "opacity_nepers": {"kwargs":{"scaleoffset": 4}},
        "excess_path_m": {"kwargs":{"scaleoffset": 4}},
    }

    for key in output_config.keys():
        f.create_dataset(key, 
                         data=spectra_data[key],
                         dtype="f", 
                         compression="gzip", 
                         compression_opts=9)
    
