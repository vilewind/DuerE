"""
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
"""
import os
import subprocess
from pathlib import Path

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

cwd = Path(__file__).parent


def get_cuda_bare_metal_version(cuda_dir):
    '''get_cuda_bare_metal_version'''
    raw_output = subprocess.check_output([cuda_dir + '/bin/nvcc', '-V'], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index('release') + 1
    release = output[release_idx].split('.')
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def append_nvcc_threads(nvcc_extra_args):
    '''append_nvcc_threads'''
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        nvcc_threads = os.getenv('NVCC_THREADS', '8')
        return nvcc_extra_args + ['--threads', nvcc_threads]
    return nvcc_extra_args


cc_flag = []
cc_flag.append('-gencode')
cc_flag.append('arch=compute_70,code=sm_70')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_80,code=sm_80')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_86,code=sm_86')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_89,code=sm_89')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_90,code=sm_90')
cc_flag.append('-gencode')
cc_flag.append('arch=compute_100,code=sm_100')

setup(
    name='stream_ext',
    version='0.0.0',
    author='linsong04',
    author_email='linsong04@baidu.com',
    description='stream_kernels pytorch extension',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
    ],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='stream_kernels',
            sources=[f'{cwd}/csrc/stream_kernels.cu'],
            include_dirs=[],
            extra_compile_args={
                'cxx': ['-O3', ],
                'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + cc_flag)
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
