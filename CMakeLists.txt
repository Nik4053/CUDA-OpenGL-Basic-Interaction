cmake_minimum_required( VERSION 3.10)

project(SampleProject LANGUAGES CUDA CXX)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDEBUG)
    message("Debug build.")
else()  
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
    message("Release build.")
endif()

ADD_LIBRARY(CUDAKERNEL
        kernel.cu
        )
set_target_properties(CUDAKERNEL PROPERTIES CUDA_ARCHITECTURES "53;60;61;62;70;72")
target_compile_options(CUDAKERNEL
        PRIVATE -Xptxas=-O3 -lcudart
        )
SET(APP_EXE main)
ADD_EXECUTABLE(${APP_EXE} ${APP_EXE}.cpp)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(common/inc)
target_link_libraries(${APP_EXE} "/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/lib/x64/freeglut.lib")
target_link_libraries(${APP_EXE} "/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/lib/x64/glew64.lib")


target_link_libraries(${APP_EXE}  CUDAKERNEL)
IF (WIN32)
    target_compile_options(${APP_EXE}
        PRIVATE /std:c++17 /Wall
    )
ELSEIF(DARWIN)
    target_compile_options(${APP_EXE}
        PRIVATE -Wall -pedantic -std=c++17 -fmax-errors=5 -march=native -D_XOPEN_SOURCE=700 -lglut -lGL -lGLU -lGLEW -L"/ProgramData/NVIDIA Corporation/CUDA Samples/v11.4/common/inc" -g -lcudart
    )
ELSE()
endif()
