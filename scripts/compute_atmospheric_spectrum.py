import time
import h5py
import os
import pathlib
import logging
import argparse
import numpy as np

import maria
from maria import Quantity, Weather
from maria.site import REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("am")

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)
args = parser.parse_args()

tag = args.tag

maria.set_local_cache_dir("/home/tm12/maria/data")

### GENERATE THE SIDE VALUES ###

write_dir = f"/scratch/gpfs/SIMONSOBS/users/tm12/maria-data/atmosphere/spectra/am/{tag}"

os.makedirs(write_dir, exist_ok=True)

for region, region_data in REGIONS.iterrows():

    region_write_path = f"{write_dir}/{region}.h5"

    if os.path.exists(region_write_path):
        continue

    with h5py.File(region_write_path, "w") as f:
        ...
    
    w_zmin = Weather(region, altitude=region_data.min_altitude, diurnal=False, seasonal=False)
    upper_pressure_level = w_zmin.pressure[np.digitize(w_zmin.pressure_level.Pa, w_zmin.pressure.Pa)].hPa

    w_zmax = Weather(region, altitude=region_data.max_altitude, diurnal=False, seasonal=False)
    lower_pressure_level = w_zmax.pressure[np.digitize(w_zmax.pressure_level.Pa, w_zmax.pressure.Pa)].hPa
    lower_pressure_level = min(lower_pressure_level, upper_pressure_level - 100)

    side_pressure_level = Quantity([10, lower_pressure_level, upper_pressure_level], "hPa").pin("hPa")
    
    effective_temperatures = []
    for pressure_level in side_pressure_level:
        for quantile in [0.05, 0.95]:
            w = Weather(region, pressure_level=pressure_level, quantiles={"temperature": quantile}, diurnal=False, seasonal=False)
            effective_temperatures.append(w.effective_temperature)
    side_effective_temperature = Quantity([min(effective_temperatures), np.mean(effective_temperatures), max(effective_temperatures)], "K")
    
    side_pwv = Quantity([0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], "mm").pin("mm")
    
    el_samples = np.array([90, 60, 40, 25, 15, 10, 5])
    side_csc_el = 1 / np.sin(np.radians(el_samples))
    
    
    spec_range_list = [(1e6, 1e7, 1e5),
                       (1e7, 1e8, 1e6),
                       (1e8, 1e9, 1e7),
                       (1e9, 1e12, 1e8),
                       (1e12, 15e12, 1e10)]
    
    spec_ranges = []
    nu = np.empty(0)
    cum_n = 0
    for i, (nu_min, nu_max, nu_step) in enumerate(spec_range_list):
    
        range_nu = np.arange(nu_min, nu_max + 1e0, step=nu_step)
        spec_ranges.append({"nu_min": Quantity(nu_min, "Hz"), 
                            "nu_max": Quantity(nu_max, "Hz"), 
                            "nu_step": Quantity(nu_step, "Hz"), 
                            "n": len(range_nu),
                            "nu_slice": slice(cum_n, cum_n + len(range_nu), 1)
                           })
        cum_n += len(range_nu)
        nu = np.r_[nu, range_nu]
    
    data_shape = (
        len(side_pressure_level),
        len(side_effective_temperature),
        len(side_pwv),
        len(side_csc_el),
        sum([spec_range["n"] for spec_range in spec_ranges]),
    )
    
    print(data_shape)
    
    data = {
        "temperature_rayleigh_jeans": np.zeros(data_shape),
        "opacity": np.zeros(data_shape),
        "path_delay": np.zeros(data_shape),
    }
    
    ### GENERATE THE SPECTRA ###
    
    from tqdm import tqdm
    
    config_pbar = tqdm(total=np.prod(data_shape[:-1]))
    
    for pressure_level_index, pressure_level in enumerate(side_pressure_level):
        for effective_temperature_index, effective_temperature in enumerate(side_effective_temperature):
            for pwv_index, pwv in enumerate(side_pwv):
            
                w = Weather(region,  
                            pressure_level=pressure_level,
                            diurnal=False,
                            seasonal=False,
                            override={"pwv": pwv})
            
                w.data["levels"]["temperature"] += effective_temperature - w.effective_temperature
    
                for csc_el_index, csc_el in enumerate(side_csc_el):
    
                    config_pbar.update()
                
                    for spec_range_index, spec_range in enumerate(spec_ranges):
                
                        config_pbar.set_postfix(region=region,
                                                pressure_level=w.pressure_level,
                                                csc_el=csc_el, 
                                                pwv=pwv,
                                                effective_temperature=effective_temperature,
                                                spec_range=(spec_range["nu_min"], spec_range["nu_max"]))
                
                        spec = w.compute_am_spectrum(am_path="/home/tm12/am/am-14.0/src/am",
                                                     nu_min=spec_range["nu_min"], 
                                                     nu_max=spec_range["nu_max"], 
                                                     nu_step=spec_range["nu_step"], 
                                                     el=np.degrees(np.arcsin(1/csc_el)))
                
                        total_slice = tuple([pressure_level_index, 
                                             effective_temperature_index, 
                                             pwv_index, 
                                             csc_el_index, 
                                             spec_range["nu_slice"]])
                
                        data["temperature_rayleigh_jeans"][total_slice] = spec["temperature_rayleigh_jeans"].K_RJ
                        data["opacity"][total_slice] = spec["opacity"]
                        data["path_delay"][total_slice] = spec["path_delay"].m
        
    fields = {
        "side_pressure_level": {"values": side_pressure_level.hPa, "units": "hPa"},
        "side_effective_temperature": {"values": side_effective_temperature.K, "units": "K"},
        "side_pwv": {"values": side_pwv.mm, "units": "mm"},
        "side_csc_el": {"values": side_csc_el, "units": ""},
        "side_nu": {"values": nu, "units": "Hz"},
        "temperature_rayleigh_jeans": {"values": data["temperature_rayleigh_jeans"], "units": "K_RJ"},
        "opacity": {"values": data["opacity"], "units": ""},
        "path_delay": {"values": data["path_delay"], "units": "m"},
    }
    
    with h5py.File(region_write_path, "w") as f:
        for field in fields:
            f.create_dataset(field, data=fields[field]["values"], dtype=np.float32, compression="gzip")
            f[field].attrs["units"] = fields[field]["units"]
    


