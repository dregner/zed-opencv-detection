CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(zed-opencv-detection)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif ()

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 OLD)
    cmake_policy(SET CMP0004 OLD)
    cmake_policy(SET CMP0015 OLD)
endif (COMMAND cmake_policy)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

SET(EXECUTABLE_OUTPUT_PATH ".")
SET(SPECIAL_OS_LIBS "")

find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
#find_package(GLUT REQUIRED)
#find_package(GLEW REQUIRED)

IF (NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread" "X11")
    add_definitions(-Wno-write-strings)
ENDIF ()
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${GLEW_INCLUDE_DIRS})
#include_directories(${GLUT_INCLUDE_DIR})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
#link_directories(${GLEW_LIBRARY_DIRS})
#link_directories(${GLUT_LIBRARY_DIRS})
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

#FILE(GLOB_RECURSE SRC_FILES src/*.c*)
FILE(GLOB_RECURSE HDR_FILES include/*.h*)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)


add_executable(cpu-detection src/opencv_detection_cpu.cpp)
add_definitions(-std=c++14 -O3)
ADD_EXECUTABLE(gpu-detection src/opencv_detection_gpu.cpp)
add_definitions(-std=c++14 -O3)
ADD_EXECUTABLE(dist-detection src/opencv_detection_position.cpp)
add_definitions(-std=c++14 -O3)

cuda_add_executable(yolov5-detection ${HDR_FILES} src/yolov5.cpp src/yololayer.cu src/calibrator.cpp)
add_definitions(-std=c++14 -g -O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)


if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else ()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif ()
SET(TRT_LIBS nvinfer)

TARGET_LINK_LIBRARIES(cpu-detection ${ZED_LIBS} ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(gpu-detection ${ZED_LIBS} ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(dist-detection ${ZED_LIBS} ${OpenCV_LIBRARIES})
target_link_libraries(yolov5-detection ${TRT_LIBS} ${SPECIAL_OS_LIBS} ${ZED_LIBS} ${OPENGL_LIBRARIES}
        ${OpenCV_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

#if (INSTALL_SAMPLES)
#    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
#    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
#endif ()
