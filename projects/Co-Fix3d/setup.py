import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == '__main__':
    setup(
        name='bev_pool',
        ext_modules=[
            make_cuda_ext(
                name='bev_pool_ext',
                module='projects.Co-Fix3D-main.co-fix3d.ops.bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
            ),
            make_cuda_ext(
                name='voxel_layer',
                module='projects.Co-Fix3D-main.co-fix3d.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ],
            ),
            make_cuda_ext(
                name='roiaware_pool3d_ext',
                module='projects.Co-Fix3D-main.co-fix3d.ops.roiaware_pool3d',
                sources=[
                    'src/points_in_boxes_cpu.cpp',
                    'src/points_in_boxes_cuda.cu',
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='locatt_ops',
                module='projects.Co-Fix3D-main.co-fix3d.ops.locatt_ops',
                sources=[
                    'src/similar.cu',
                    'src/weighting.cu',
                    'src/localAttention.cpp'
                ],
            ),
            make_cuda_ext(
                name='msmv_sampling_cuda',
                module='projects.Co-Fix3D-main.co-fix3d.ops.msmv_sampling',
                sources=[
                    'src/msmv_sampling.cpp',
                    'src/msmv_sampling_forward.cu',
                    'src/msmv_sampling_backward.cu'
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
