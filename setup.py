import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="warp_pipe_ext",
    ext_modules=[
        CUDAExtension(
            name="warp_pipe_ext",
            sources=[
                "csrc/kernels/warp_pipe_r32.cu",
                "csrc/kernels/warp_pipe_r8.cu",
                "csrc/kernels/warp_pipe_r16.cu",
                "csrc/kernels/warp_pipe_r64.cu",
                "csrc/kernels/prefetch_kernel.cu",
                "csrc/bridge/scheduler_kernel_bridge.cpp",
                "csrc/dispatch/warp_pipe_dispatcher.cpp",
                "csrc/dispatch/segment_builder.cpp",
                "csrc/pybind/warp_pipe_bindings.cpp",
            ],
            include_dirs=[
                os.path.join(THIS_DIR, "csrc"),
                os.path.join(THIS_DIR, "csrc/kernels"),
                os.path.join(THIS_DIR, "csrc/bridge"),
                os.path.join(THIS_DIR, "csrc/dispatch"),
                os.path.join(THIS_DIR, "csrc/memory"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "-gencode",
                    "arch=compute_86,code=sm_86",
                    "--use_fast_math",
                    "-Xptxas",
                    "-O3",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
