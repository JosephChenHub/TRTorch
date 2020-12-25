
cmake_minimum_required(VERSION 3.10)
project(TRTorch LANGUAGES CXX)

# lib's name
set(EXE_NAME trtorch)  

# use c++14 
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

# Enable compiler warnings
if (CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -ggdb -g -O0 -Wno-deprecated-declarations -fPIC -pthread")
    else(CMAKE_BUILD_TYPE MATCHES "Release")
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-deprecated-declarations -Wno-unused-function -fPIC -pthread")
    endif()
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------
# dependencies 
# -----------------------------------------

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_MYELIN myelin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
#find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
#if(NOT TENSORRT_FOUND)
#    message(ERROR "Cannot find TensorRT library.")
#endif()

# CUDA
find_package(CUDA 10.0 REQUIRED)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11 -g -G")
else(CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda -std=c++11 -O3")
endif()
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})

# libtorch 
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Torch libs:${TORCH_LIBRARIES}")
message(STATUS "CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")
message(STATUS "Find Torch VERSION: ${Torch_VERSION}")
add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})



# -----------------------------------------
#  compilation 
# -----------------------------------------
include_directories(${PROJECT_SOURCE_DIR})
set(CXX_SRC  ${PROJECT_SOURCE_DIR}/core/compiler.cpp)

set(COMMON_INC ${CUDA_INCLUDE_DIRS}  ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS}) 


list(APPEND SUBSETS conversion lowering util runtime)

foreach(SUBMOD ${SUBSETS})
   add_subdirectory(${PROJECT_SOURCE_DIR}/core/${SUBMOD}) 
   add_library(${SUBMOD} SHARED ${SUB_SRCS} )
   target_link_libraries(${SUBMOD} PUBLIC ${TENSORRT_LIBRARY} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
   target_include_directories(${SUBMOD} PUBLIC  ${COMMON_INC}   ${SUB_INC} ${PROJECT_SOURCE_DIR})

   #list(APPEND CORE_SRC ${SUB_SRCS})
   #list(APPEND CORE_INC ${SUB_INC})
endforeach()

add_library(core SHARED ${CXX_SRC} )
target_link_libraries(core PUBLIC conversion lowering util runtime) 


#add_library(core SHARED ${CORE_SRC} ${CXX_SRC})
#target_include_directories(core PUBLIC ${CORE_INC} ${COMMON_INC} {PROJECT_SOURCE_DIR})
#target_link_libraries(core PUBLIC ${TENSORRT_LIBRARY} ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})





add_subdirectory(cpp/api) 
add_library(${EXE_NAME} SHARED ${SUB_SRCS})
target_link_libraries(${EXE_NAME} PUBLIC core) 
target_include_directories(${EXE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/cpp/api/${SUB_INC} ${PROJECT_SOURCE_DIR}/core/)




