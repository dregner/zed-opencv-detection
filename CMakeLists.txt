CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(zed-opencv-detection)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(cpu-detection src/opencv_detection_cpu.cpp)
add_definitions(-std=c++14 -O3)
ADD_EXECUTABLE(gpu-detection src/opencv_detection_gpu.cpp)
add_definitions(-std=c++14 -O3)
ADD_EXECUTABLE(dist-detection src/opencv_detection_position.cpp)
add_definitions(-std=c++14 -O3)


if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES(cpu-detection ${ZED_LIBS} ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(gpu-detection ${ZED_LIBS} ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(dist-detection ${ZED_LIBS} ${OpenCV_LIBRARIES})


if(INSTALL_SAMPLES)
    LIST(APPEND SAMPLE_LIST ${PROJECT_NAME})
    SET(SAMPLE_LIST "${SAMPLE_LIST}" PARENT_SCOPE)
endif()
