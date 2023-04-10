cmake_minimum_required(VERSION 3.10) 

project(CF_CPU VERSION 1.0)

set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED True) 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_definitions(-DSPLATT_VAL_TYPEWIDTH=32)

find_package(OpenMP)
if(${OPENMP_FOUND})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message("OpenMP not found! ")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src 
                    ${CMAKE_SOURCE_DIR}/src/splatt 
                    ${CMAKE_SOURCE_DIR}/src/modules 
                    ${CMAKE_SOURCE_DIR}/src/modules/memory 
                    ${CMAKE_SOURCE_DIR}/src/modules/random 
                    ) 

file(GLOB CF_SOURCES 
    # ${CMAKE_SOURCE_DIR}/src/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/splatt/*.c 
    ${CMAKE_SOURCE_DIR}/src/modules/behavior_aggregators/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/modules/datasets/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/modules/embeddings/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/modules/models/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/modules/negative_samplers/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/modules/optimizers/*.cpp 
    ${CMAKE_SOURCE_DIR}/src/modules/train/*.cpp 
    )

find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIR extern/eigen)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(cf ${CF_SOURCES})

add_subdirectory(extern/pybind11)
file(GLOB PYBIND_SOURCES src/pybind/*.cpp)
pybind11_add_module(cf_c MODULE ${PYBIND_SOURCES})
target_link_libraries(cf_c PUBLIC cf)

set_target_properties(cf_c PROPERTIES CXX_VISIBILITY_PRESET "hidden" 
                          CUDA_VISIBILITY_PRESET "hidden")
