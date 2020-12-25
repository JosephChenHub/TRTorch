import os
import sys
import glob
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from distutils.cmd import Command
from wheel.bdist_wheel import bdist_wheel

from torch.utils import cpp_extension
from shutil import copyfile, rmtree

import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))

__version__ = '0.2.0a0'

CXX11_ABI = False

if "--use-cxx11-abi" in sys.argv:
    sys.argv.remove("--use-cxx11-abi")
    CXX11_ABI = True


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

CMAKE_PREFIX = '/workspace/TRTorch/libtorch/;/workspace/TensorRT-7.0.0.11/'

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DEXAMPLE_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
            "-DCMAKE_PREFIX_PATH={}".format(CMAKE_PREFIX)
        ]
        build_args = []

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            try:
                import ninja
                if not cmake_generator:
                    cmake_args += ["-GNinja"]
            except:
                pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def build_libtrtorch_pre_cxx11_abi(develop=True, use_dist_dir=True, cxx11_abi=False):
    pass


def gen_version_file():
    if not os.path.exists(dir_path + '/trtorch/_version.py'):
        os.mknod(dir_path + '/trtorch/_version.py')

    with open(dir_path + '/trtorch/_version.py', 'w') as f:
        print("creating version file")
        f.write("__version__ = \"" + __version__ + '\"')


def copy_libtrtorch(multilinux=False):
    if not os.path.exists(dir_path + '/trtorch/lib'):
        os.makedirs(dir_path + '/trtorch/lib')

    print("copying library into module")
    copyfile(os.path.join(dir_path, "../build/libtrtorch.so"), dir_path + '/trtorch/lib/libtrtorch.so')


class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        global CXX11_ABI
        build_libtrtorch_pre_cxx11_abi(develop=True, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtrtorch()
        develop.run(self)


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global CXX11_ABI
        build_libtrtorch_pre_cxx11_abi(develop=False, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtrtorch()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        global CXX11_ABI
        build_libtrtorch_pre_cxx11_abi(develop=False, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtrtorch()
        bdist_wheel.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = [
        './build', './dist', './trtorch/__pycache__', './trtorch/lib', './*.pyc', './*.tgz', './*.egg-info'
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print('Removing %s' % os.path.relpath(path))
                rmtree(path)


ext_modules = [
    cpp_extension.CUDAExtension(
        'trtorch._C', [
            'trtorch/csrc/trtorch_py.cpp',
            'trtorch/csrc/tensorrt_backend.cpp',
            'trtorch/csrc/tensorrt_classes.cpp',
            'trtorch/csrc/register_tensorrt_classes.cpp',
        ],
        library_dirs=[(dir_path + '/trtorch/lib/'), "/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/"], 
        libraries=["trtorch"],
        include_dirs=[
            dir_path + "trtorch/csrc",
            dir_path + "/../",
            dir_path + "/tensorrt/include",
        ],
        extra_compile_args=[
            "-Wno-deprecated",
            "-Wno-deprecated-declarations",
        ] + (["-D_GLIBCXX_USE_CXX11_ABI=1"] if CXX11_ABI else ["-D_GLIBCXX_USE_CXX11_ABI=0"]),
        extra_link_args=[
            "-Wno-deprecated", "-Wno-deprecated-declarations", "-Wl,--no-as-needed", "-ltrtorch",
            "-Wl,-rpath,$ORIGIN/lib", "-lpthread", "-ldl", "-lutil", "-lrt", "-lm", "-Xlinker", "-export-dynamic"
        ] + (["-D_GLIBCXX_USE_CXX11_ABI=1"] if CXX11_ABI else ["-D_GLIBCXX_USE_CXX11_ABI=0"]),
        undef_macros=["NDEBUG"])
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='trtorch',
      version=__version__,
      author='NVIDIA',
      author_email='narens@nvidia.com',
      url='https://nvidia.github.io/TRTorch',
      description='A compiler backend for PyTorch JIT targeting NVIDIA GPUs',
      long_description_content_type='text/markdown',
      long_description=long_description,
      ext_modules=ext_modules,
      #ext_modules=[CMakeExtension("trtorch/csrc")],
      install_requires=[
          'torch==1.6.0',
      ],
      setup_requires=[],
      cmdclass={
          'install': InstallCommand,
          'clean': CleanCommand,
          'develop': DevelopCommand,
          'build_ext': cpp_extension.BuildExtension,
          #'build_ext': CMakeBuild, 
          'bdist_wheel': BdistCommand,
      },
      zip_safe=False,
      license="BSD",
      packages=find_packages(),
      classifiers=[
          "Development Status :: 4 - Beta", "Environment :: GPU :: NVIDIA CUDA",
          "License :: OSI Approved :: BSD License", "Intended Audience :: Developers",
          "Intended Audience :: Science/Research", "Operating System :: POSIX :: Linux", "Programming Language :: C++",
          "Programming Language :: Python", "Programming Language :: Python :: Implementation :: CPython",
          "Topic :: Scientific/Engineering", "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Software Development", "Topic :: Software Development :: Libraries"
      ],
      python_requires='>=3.6',
      include_package_data=True,
      package_data={
          'trtorch': ['lib/*.so'],
      },
      exclude_package_data={
          '': ['*.cpp', '*.h'],
          'trtorch': ['csrc/*.cpp'],
      })
