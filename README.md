# TRTorch

[![Documentation](https://img.shields.io/badge/docs-master-brightgreen)](https://nvidia.github.io/TRTorch/)

> Ahead of Time (AOT) compiling for PyTorch JIT

TRTorch is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch's Just-In-Time (JIT) compiler, TRTorch is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting a TensorRT engine. TRTorch operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly. After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT's suite of configurations at compile time, so you are able to specify operating precision (FP32/FP16/INT8) and other settings for your module.

More Information / System Architecture:

- [GTC 2020 Talk](https://developer.nvidia.com/gtc/2020/video/s21671)

## Example Usage

### C++
```c++
#include "torch/script.h"
#include "trtorch/trtorch.h"

...
auto compile_settings = trtorch::CompileSpec(dims);
// FP16 execution
compile_settings.op_precision = torch::kFloat;
// Compile module
auto trt_mod = trtorch::CompileGraph(ts_mod, compile_settings);
// Run like normal
auto results = trt_mod.forward({in_tensor});
// Save module for later
trt_mod.save("trt_torchscript_module.ts");
...
```

### Python
```py
import trtorch

...
compile_settings = {
    "input_shapes": [
        {
            "min": [1, 3, 224, 224],
            "opt": [1, 3, 512, 512],
            "max": [1, 3, 1024, 1024]
        }, # For static size [1, 3, 224, 224]
    ],
    "op_precision": torch.half # Run with FP16
}

trt_ts_module = trtorch.compile(torch_script_module, compile_settings)

input_data = input_data.half()
result = trt_ts_module(input_data)
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts")
```

> Notes on running in lower precisions:
> - Set precision with compile_spec.op_precision
> - The module should be left in FP32 before compilation (FP16 can support half tensor models)
> - In FP16 only input tensors should be converted to FP16, other precisions use FP32

## Platform Support

| Platform | Support |
| -------- | ------- |
| Linux AMD64 / GPU   | **Supported** |
| Linux aarch64 / GPU | **Native Compilation Supported on JetPack-4.4** |
| Linux aarch64 / DLA | **Native Compilation Supported on JetPack-4.4** |
| Windows / GPU       | **Unofficial Support** |
| Linux ppc64le / GPU | - |

> Note: Refer NVIDIA NGC container(https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) for PyTorch libraries on JetPack.

### Dependencies
These are the following dependencies used to verify the testcases. TRTorch can work with other versions, but the tests are not guaranteed to pass.

- Libtorch 1.6.0 (built with CUDA 10.2)
- CUDA 10.2
- cuDNN 7
- TensorRT 7.0.0.11 or higher 

## Compiling TRTorch (use CMake)
1. download the TensorRT library (e.g., TensorRT-7.0.0.11 with cuda 10.2 support) 
2. download  libtorch built with Pre-Cxx ABI: 
```
wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.6.0.zip
```
or 
```
axel -n 5 "https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.6.0.zip"
```
3. build the library `trtorch`:
```
mkdir build && cd build \
cmake .. -DCMAKE_PREFIX_PATH="${TENSORRT_PATH}; ${LIBTORCH_PATH}" \
make -j8
```
where `${TENSORRT_PATH}` and `${LIBTORCH_PATH}` refer to the path that contains TensorRT and libtorch respectively. 
4. build the python wrapper:
```
cd py \
python3 setup.py install 
```
if you have no access to `root`, use the command :
```
python3 setup.py install --prefix which_dir_you_want_install 
```
remember add the path `which_dir_you_want_install` to the python environment.

5. run the `demo.py`:
```
python3 demo.py 
```


## Compiling TRTorch (use Bazel)
see the `master` branch


## How do I add support for a new op...

### In TRTorch?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/NVIDIA/TRTorch/issues) for information on the support status of various operators.

### In my application?

> The Node Converter Registry is not exposed in the top level API but in the internal headers shipped with the tarball.

You can register a converter for your op using the `NodeConverterRegistry` inside your application.

## Structure of the repo

| Component     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| [**core**](core)  | Main JIT ingest, lowering, conversion and execution implementations |
| [**cpp**](cpp)   | C++ specific components including API and example applications |
| [**cpp/api**](cpp/api)   | C++ API for TRTorch |
| [**py**](py)   | Python API for TRTorch |
| [**tests**](tests) | Unit test for TRTorch |

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The TRTorch license can be found in the LICENSE file. It is licensed with a BSD Style licence
