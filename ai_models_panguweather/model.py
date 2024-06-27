# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import onnxruntime as ort
import datetime
from netCDF4 import Dataset as DS
from ai_models.model import Model

LOG = logging.getLogger(__name__)


class PanguWeather(Model):
    # Download
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/pangu-weather/{file}"
    )
    download_files = ["pangu_weather_24.onnx", "pangu_weather_6.onnx"]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ["msl", "10u", "10v", "2t"]
    param_level_pl = (
        ["z", "q", "t", "u", "v"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    # Output
    expver = "pguw"

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads

    def run(self):
        fields_pl = self.fields_pl

        param, level = self.param_level_pl
        fields_pl = fields_pl.sel(param=param, level=level)
        fields_pl = fields_pl.order_by(param=param, level=level)

        fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
        fields_pl_numpy = fields_pl_numpy.reshape((5, 13, 721, 1440))

        fields_sfc = self.fields_sfc
        fields_sfc = fields_sfc.sel(param=self.param_sfc)
        fields_sfc = fields_sfc.order_by(param=self.param_sfc)

        fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)

        input = fields_pl_numpy
        input_surface = fields_sfc_numpy

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.num_threads

        pangu_weather_24 = os.path.join(self.assets, "pangu_weather_24.onnx")
        pangu_weather_6 = os.path.join(self.assets, "pangu_weather_6.onnx")

        # That will trigger a FileNotFoundError

        os.stat(pangu_weather_24)
        os.stat(pangu_weather_6)

        with self.timer(f"Loading {pangu_weather_24}"):
            ort_session_24 = ort.InferenceSession(
                pangu_weather_24,
                sess_options=options,
                providers=self.providers,
            )

        with self.timer(f"Loading {pangu_weather_6}"):
            ort_session_6 = ort.InferenceSession(
                pangu_weather_6,
                sess_options=options,
                providers=self.providers,
            )

        self.write_input_fields(fields_pl + fields_sfc)

        input_24, input_surface_24 = input, input_surface

        #Create dictionary to hold output and variable mappings
        if 'n' in self.nc_or_grib:
            out,mapping,varmap = initialize_nc_dict(self.lead_time)

        #Save initial conditions to output dictionary and write to grib
        outputpl0 = fields_pl_numpy.reshape((-1, 721, 1440))
        for data, fs in zip(outputpl0, fields_pl):
            if 'n' in self.nc_or_grib:
                shortname = fs.handle.get("shortName")
                mappedvar = varmap[shortname]
                level = fs.handle.get("level")
                if level!=0:
                    levelidx = mapping[level]
                    out[mappedvar]['values'][0,levelidx,:,:] = data
                else:
                    out[mappedvar]['values'][0,:,:] = data
            if 'g' in self.nc_or_grib:
                self.write(data, template=fs, step=0)

        outputsfc0 = fields_sfc_numpy.reshape((-1, 721, 1440))
        for data, fs in zip(outputsfc0, fields_sfc):
            if 'n' in self.nc_or_grib:
                shortname = fs.handle.get("shortName")
                mappedvar = varmap[shortname]
                level = fs.handle.get("level")
                if level!=0:
                    levelidx = mapping[level]
                    out[mappedvar]['values'][0,levelidx,:,:] = data
                else:
                    out[mappedvar]['values'][0,:,:] = data
            if 'g' in self.nc_or_grib:
                self.write(data, template=fs, step=0)

        with self.stepper(6) as stepper:
            for i in range(self.lead_time // 6):
                step = (i + 1) * 6

                if (i + 1) % 4 == 0:
                    output, output_surface = ort_session_24.run(
                        None,
                        {
                            "input": input_24,
                            "input_surface": input_surface_24,
                        },
                    )
                    input_24, input_surface_24 = output, output_surface
                else:
                    output, output_surface = ort_session_6.run(
                        None,
                        {
                            "input": input,
                            "input_surface": input_surface,
                        },
                    )
                input, input_surface = output, output_surface

                # Save the results to output dictionary and write to grib

                pl_data = output.reshape((-1, 721, 1440))

                for data, fs in zip(pl_data, fields_pl):
                    if 'n' in self.nc_or_grib:
                        shortname = fs.handle.get("shortName")
                        mappedvar = varmap[shortname]
                        level = fs.handle.get("level")
                        if level!=0:
                            levelidx = mapping[level]
                            out[mappedvar]['values'][i+1,levelidx,:,:] = data
                        else:
                            out[mappedvar]['values'][i+1,:,:] = data
                    if 'g' in self.nc_or_grib:
                        self.write(data, template=fs, step=step)

                sfc_data = output_surface.reshape((-1, 721, 1440))
                for data, fs in zip(sfc_data, fields_sfc):
                    if 'n' in self.nc_or_grib:
                        shortname = fs.handle.get("shortName")
                        mappedvar = varmap[shortname]
                        level = fs.handle.get("level")
                        if level!=0:
                            levelidx = mapping[level]
                            out[mappedvar]['values'][i+1,levelidx,:,:] = data
                        else:
                            out[mappedvar]['values'][i+1,:,:] = data
                    if 'g' in self.nc_or_grib:
                        self.write(data, template=fs, step=step)

                stepper(i, step)
    
        #Save output to nc
        if 'n' in self.nc_or_grib:
            write_nc(out,self.lead_time,self.date,self.time,self.ncpath)

def create_variable(f, name, dimensions, data, attrs):
    if name in ['time','level']:
        dtype = 'i4'
    else:
        dtype = 'f4'
    var = f.createVariable(name, dtype, dimensions,compression='zlib',complevel=4)
    var[:] = data
    for attr_name, attr_value in attrs.items():
        var.setncattr(attr_name, attr_value)

def initialize_nc_dict(lead_time):
    hour_steps = 6
    out = {
        'u10': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '10 metre U wind component', 'units': 'm s-1'
        },
        'v10': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '10 metre V wind component', 'units': 'm s-1'
        },
        't2': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '2 metre temperature', 'units': 'K'
        },
        'msl': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': 'Pressure reduced to MSL', 'units': 'Pa'
        },
        't': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'Temperature', 'units': 'K'
        },
        'u': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'U component of wind', 'units': 'm s-1'
        },
        'v': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'V component of wind', 'units': 'm s-1'
        },
        'z': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'Geopotential', 'units': 'm2 s-2'
        },
        'q': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'Relative humidity', 'units': '%'
        },
    }

    mapping = {
        50:12,
        100:11,
        150:10,
        200:9,
        250:8,
        300:7,
        400:6,
        500:5,
        600:4,
        700:3,
        850:2,
        925:1,
        1000:0
    }

    varmap = {
        "u":"u",
        "v":"v",
        "z":"z",
        "t":"t",
        "q":"q",
        "10u":"u10",
        "10v":"v10",
        "msl":"msl",
        "2t":"t2"
    }


    return out,mapping,varmap

def write_nc(out,lead_time,date,time,path):
    hour_steps = 6
    outdir = path
    f = DS(outdir, 'w', format='NETCDF4')
    f.createDimension('time', lead_time // hour_steps + 1)
    f.createDimension('level', 13)
    f.createDimension('longitude', 1440)
    f.createDimension('latitude', 721)

    year = str(date)[0:4]
    month = str(date)[4:6]
    day = str(date)[6:8]
    hh = str(time)[0:2]
    initdt = datetime.datetime.strptime(f"{year}{month}{day}{hh}","%Y%m%d%H")
    inityr = str(initdt.year)
    initmnth = str(initdt.month).zfill(2)
    initday = str(initdt.day).zfill(2)
    inithr = str(initdt.hour).zfill(2)
    times = []
    for i in np.arange(0,lead_time + hour_steps,hour_steps):
        times.append(int((initdt + datetime.timedelta(hours=int(i))).timestamp()))

    # Create time, longitude, latitude, and level variables in the NetCDF file
    create_variable(
        f, 'time', ('time',), np.array(times), {
            'long_name': 'Date and Time', 'units': 'seconds since 1970-1-1',
            'calendar': 'standard'
        }
    )
    create_variable(
        f, 'longitude', ('longitude',), np.arange(0, 360, 0.25), {
            'long_name': 'Longitude', 'units': 'degree'
        }
    )
    create_variable(
        f, 'latitude', ('latitude',), np.arange(-90, 90.25, 0.25)[::-1], {
            'long_name': 'Latitude', 'units': 'degree'
        }
    )
    create_variable(
        f, 'level', ('level',), np.array(
            [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        )[::-1], {'long_name': 'Isobaric surfaces', 'units': 'hPa'}
    )

    # Create variables for each meteorological parameter
    for variable in [
        'u10', 'v10', 't2', 'msl', 't', 'u', 'v', 'z', 'q'
    ]:
        dims = ('time', 'level', 'latitude', 'longitude') if variable in [
            'u', 'v', 'z', 't', 'q'
        ] else ('time', 'latitude', 'longitude')
        create_variable(
            f, variable, dims, out[variable]['values'], {
                'long_name': out[variable]['name'], 'units': out[variable]['units']
            }
        )

    f.Conventions = 'CF-1.8'
    f.model_name = 'PanguWeather'
    f.model_version = 'v1'
    f.initialization_model = 'GFS'
    f.initialization_time = '%s-%s-%sT%s:00:00' % (inityr,initmnth,initday,inithr)
    f.first_forecast_hour = str(0)
    f.last_forecast_hour = str(lead_time)
    f.forecast_hour_step = str(6)
    f.creation_time = (datetime.datetime.utcnow()).strftime('%Y-%m-%dT%H:%M:%S')
    f.close()
