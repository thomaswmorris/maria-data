import argparse
import h5py
import logging
import pathlib
import re
import subprocess
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("am")

parser = argparse.ArgumentParser()
parser.add_argument("--region", type=str, help="maria region")
parser.add_argument("--res", type=str, help="Resolution mode, ('low', 'medium', or 'high').")
args = parser.parse_args()

region = args.region

# in GHz
f_step = {"low": 1e0, "medium": 1e-1, "high": 1e-2}[args.res]

# for running on della
MARIA_DATA_PATH = "/users/tom/maria/data"
AM_PATH = "/users/tom/am-14.0/src/am"

write_path = f"{MARIA_DATA_PATH}/atmosphere/spectra/am/{args.res}-res/{region}.h5"
regions = pd.read_csv(f"{MARIA_DATA_PATH}/regions.csv", index_col=0)

def get_saturation_pressure(air_temp): # units are (°K, %)
    T = air_temp - 273.15 # in °C
    a, b, c = 611.21, 17.67, 238.88 # units are Pa, ., °C
    return a * np.exp(b * T / (c + T))

def relative_to_absolute_humidity(rel_hum, air_temp):
    return 1e-2 * rel_hum * get_saturation_pressure(air_temp) / (461.5 * air_temp)

quantiles = {}
spectra_data = {}
durations = []

with h5py.File(f"{MARIA_DATA_PATH}/atmosphere/weather/era5/{region}.h5", "r") as f:
    fields = list(f["data"].keys())

    pressure_profile = 1e2 * f["pressure_levels"][:] # in Pa
    quantile_levels = f["quantile_levels"][:]

    for attr in fields:
        quantiles[attr] = f["data"][attr]["normalized_quantiles"][:] * f["data"][attr]["scale"][:] + f["data"][attr]["mean"][:]

geopotential_profile = np.median(quantiles["geopotential"], axis=(0,1))[7]
temperature_profile  = np.median(quantiles["temperature"], axis=(0,1))[7]
humidity_profile     = np.median(quantiles["humidity"], axis=(0,1))[7]
ozone_profile        = np.median(quantiles["ozone"], axis=(0,1))[7] * 28.97 / 48 # convert to vmr

absolute_humidity_profile = relative_to_absolute_humidity(humidity_profile, temperature_profile)
typical_pwv = np.trapezoid(relative_to_absolute_humidity(humidity_profile, temperature_profile), geopotential_profile/9.80665)
pwv_per_layer_mm = typical_pwv * absolute_humidity_profile / absolute_humidity_profile.sum()

logger.info(f"typical PWV for {region}: {typical_pwv} mm")

zenith_pwv_samples = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
zenith_pwv_samples = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

pwv_scales = zenith_pwv_samples / typical_pwv

ground_temperatures = np.percentile(quantiles["temperature"][..., 0], q=[0, 50, 100])
elevation_samples = np.linspace(5, 90, 18)
elevation_samples = np.linspace(10, 90, 9)

f_min  = f_step
f_max  = 1e3
nu_GHz = np.arange(f_min, f_max + f_step, f_step)

TRJ = np.zeros((
                len(pwv_scales), 
                len(ground_temperatures),
                len(elevation_samples),
                len(nu_GHz)))
TAU = np.zeros(TRJ.shape)
L = np.zeros(TRJ.shape)

logger.info("starting grid")

for i_ws, ws in enumerate(pwv_scales):
    for i_gt, gt in enumerate(ground_temperatures):
        for i_el, el in enumerate(elevation_samples):

            start_time = time.time()

            _temperature_profile = gt + temperature_profile - temperature_profile[0]

            text  = f"f {f_min} GHz {f_max} GHz {f_step} GHz"
            text += f"\noutput f GHz Trj K tau neper L m"
            text += f"\ntol 0"
            text += f"\nNscale h2o {ws}"
            text += f"\nza {90 - el} deg"
            text += f"\nT0 0 K"

            for i in np.arange(len(pressure_profile))[::-1]:

                text += f"\n\nlayer"
                text += f"\nPbase {pressure_profile[i]:.01f} Pa"
                text += f"\nTbase {_temperature_profile[i]:.01f} K"
                text += f"\ncolumn h2o {1e3 * pwv_per_layer_mm[i]:.03e} um_pwv"
                text += f"\ncolumn o3 vmr {ozone_profile[i]:.03e}"
                text += f"\ncolumn dry_air vmr"

            pathlib.Path(f"/tmp/am/{region}").mkdir(parents=True, exist_ok=True)
            config_path = f"/tmp/am/{region}/config.amc"
            with open(config_path, "w") as f:
                f.write(text)


            proc = subprocess.run([AM_PATH, config_path], capture_output=True, text=True)
            durations.append(time.time() - start_time)

            spec = pd.DataFrame(np.array(proc.stdout.split()).reshape(-1,4).astype(float), columns=["nu", "trj", "tau", "L"])
            zpwvstr, lospwvstr = re.findall(r"# *\((.+) um_pwv\) *\((.+) um_pwv\) *", proc.stderr)[0]
            zpwv, lospwv = float(zpwvstr), float(lospwvstr)

            TAU[i_ws, i_gt, i_el] = spec.tau
            TRJ[i_ws, i_gt, i_el] = spec.trj
            L[i_ws, i_gt, i_el] = spec.L

            index_90GHz = np.argmin(np.abs(nu_GHz - 90))

            logger.info(f"{region} | n_h2o = {ws:.02f} | w_z={1e-3 * zpwv:.02f}mm | w={1e-3 * lospwv:.02f}mm | ground_temp={gt:.1f}K | el={el}deg")


spectra_data["side_nu_GHz"] = np.array(nu_GHz)
spectra_data["side_elevation_deg"] = np.array(elevation_samples)
spectra_data["side_base_temperature_K"] = np.array(ground_temperatures)
spectra_data["side_zenith_pwv_mm"] = np.array(zenith_pwv_samples)
spectra_data["brightness_temperature_rayleigh_jeans_K"] = TRJ
spectra_data["opacity_nepers"] = TAU
spectra_data["excess_path_m"] = L


with h5py.File(write_path, "w") as f:

    for key in ["side_nu_GHz", "side_zenith_pwv_mm", "side_base_temperature_K", "side_elevation_deg"]:

        f.create_dataset(key, data=spectra_data[key].astype(float), dtype="f")

    output_config = {
        "brightness_temperature_rayleigh_jeans_K": {"kwargs":{"scaleoffset": 4}},
        "opacity_nepers": {"kwargs":{"scaleoffset": 4}},
        "excess_path_m": {"kwargs":{"scaleoffset": 4}},
    }

    for key in output_config.keys():

        f.create_group(key)

        offset = spectra_data[key].mean(axis=-1)[..., None]
        scale = spectra_data[key].std(axis=-1)[..., None]
        relative = (spectra_data[key] - offset) / scale

        f[key].create_dataset("offset", data=offset)
        f[key].create_dataset("scale", data=scale)
        f[key].create_dataset("relative", data=relative, 
                                 dtype="f", 
                                 compression="gzip", 
                                 compression_opts=9, 
                                 **output_config[key]["kwargs"],
                            )

