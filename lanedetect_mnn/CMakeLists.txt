cmake_minimum_required(VERSION 3.1)
project(lane)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/lane.cpp)
set(SRC_MAIN ${PROJECT_SOURCE_DIR}/main.cpp)
set(MNN_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
set(SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
# set(INCLUDE_DIR2 ${PROJECT_SOURCE_DIR}/include/MNN)
set(MNN_LIB ${PROJECT_SOURCE_DIR}/lib/libMNN.dylib)
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -O0 -std=c++11 ")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")
include_directories(${MNN_INCLUDE_DIR} ${SRC_INCLUDE_DIR})

add_executable(lane ${SRC_DIR} ${OpenCV_INCLUDE_DIRS} ${SRC_MAIN})
target_link_libraries(lane ${OpenCV_LIBRARIES} ${MNN_LIB})