#!/usr/bin/env python
# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import io
import os
import platform
import subprocess
import sys

import setuptools


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(file_path, encoding="utf-8").read()


version = "0.0.1"

assert version


def check_gpus():
    try:
        n = 0
        for line in subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
        ).split("\n"):
            if line.startswith("GPU"):
                n += 1
        return n
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def has_gpu():
    return check_gpus() > 0


onnxruntime = "onnxruntime"
if sys.platform == "darwin":
    if platform.machine() == "arm64":
        onnxruntime = "onnxruntime-silicon"

if has_gpu():
    onnxruntime = "onnxruntime-gpu"


setuptools.setup(
    name="ai-models-panguweather-gfs",
    # python_requires="<3.11",  # For now, does not support Python 3.11
    version=version,
    description="Run panguweather with capabilities for GFS and GDAS initial conditions and NetCDF output",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Jacob Radford",
    author_email="jacob.t.radford@gmail.com",
    license="Apache License Version 2.0",
    url="https://github.com/jacob-radford/ai-models-panguweather-gfs",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=["GPUtil"],
    install_requires=[
        "onnx",
        os.environ.get("ONNXRUNTIME", onnxruntime),
    ],
    zip_safe=True,
    keywords="tool",
    entry_points={
        "ai_models_gfs.model": [
            "panguweather = ai_models_panguweather_gfs.model:PanguWeather",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
    ],
)
