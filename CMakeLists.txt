cmake_minimum_required(VERSION 3.20)
project(optimization_learning)

option(BUILD_BENCHMARKING "Build benchmarks" ON)

SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
#      set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
# endif()

find_package(glog REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED COMPONENTS
  common io search visualization filters features registration)
find_package(Ceres REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)

add_subdirectory(thirdparty/small_gicp)
add_subdirectory(thirdparty/ndt_omp)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  thirdparty/small_gicp/include
  thirdparty/ndt_omp/include
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

message("Eigen version: ${EIGEN3_VERSION_STRING}")
message("GTSAM version: ${GTSAM_VERSION}")
message("Ceres version: ${Ceres_VERSION}")
message("PCL version: ${PCL_VERSION}")
message("Glog version: ${glog_VERSION}")

add_executable(R_mean src/R_mean.cpp)
ament_target_dependencies(R_mean
  rclcpp
)
target_link_libraries(R_mean
  ${PCL_LIBRARIES}
  Ceres::ceres
  glog::glog
  gtsam
)

add_executable(icp src/icp.cpp)
ament_target_dependencies(icp
  rclcpp
)
target_link_libraries(icp
  ${PCL_LIBRARIES}
  Ceres::ceres
  glog::glog
  gtsam
)

add_executable(odometry_node src/odometry.cpp)
target_link_libraries(odometry_node
  ${PCL_LIBRARIES}
  Ceres::ceres
  glog::glog
  gtsam
  ndt_omp
  small_gicp
)

ament_target_dependencies(odometry_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  pcl_conversions
  tf2
  tf2_ros
)

install(TARGETS R_mean icp odometry_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

if (BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(icp_test test/icp_test.cpp)
  ament_target_dependencies(icp_test
    rclcpp
  )
  target_link_libraries(icp_test
    ${PCL_LIBRARIES}
    Ceres::ceres
    glog::glog
    gtsam
  )
  ament_add_gtest(registration_test test/registration_test.cpp)
  ament_target_dependencies(registration_test
    rclcpp
  )
  target_link_libraries(registration_test
    ${PCL_LIBRARIES}
    Ceres::ceres
    glog::glog
    gtsam
    small_gicp
    ndt_omp
  )

  ament_add_gtest(downsampling_test test/downsampling_test.cpp)
  ament_target_dependencies(downsampling_test
    rclcpp
  )
  target_link_libraries(downsampling_test
    ${PCL_LIBRARIES}
    glog::glog
    small_gicp
  )

  ament_add_gtest(transformcloud_test test/transformcloud_test.cpp)
  ament_target_dependencies(transformcloud_test
    rclcpp
  )
  target_link_libraries(transformcloud_test
    ${PCL_LIBRARIES}
    glog::glog
  )
endif()

if (BUILD_BENCHMARKING)
  find_package(benchmark REQUIRED)
  add_executable(registration_benchmark benchmark/registration_benchmark.cpp)
  target_link_libraries(registration_benchmark
    ${PCL_LIBRARIES}
    Ceres::ceres
    glog::glog
    gtsam
    small_gicp
    ndt_omp
    benchmark::benchmark
  )

  add_executable(parallel_benchmark benchmark/parallel_benchmark.cpp)
  target_link_libraries(parallel_benchmark
    ${PCL_LIBRARIES}
    Ceres::ceres
    glog::glog
    gtsam
    small_gicp
    ndt_omp
    benchmark::benchmark
  )

  add_executable(downsampling_benchmark benchmark/downsampling_benchmark.cpp)
  target_link_libraries(downsampling_benchmark
    ${PCL_LIBRARIES}
    glog::glog
    small_gicp
    benchmark::benchmark
  )

  add_executable(transformcloud_benchmark benchmark/transformcloud_benchmark.cpp)
  target_link_libraries(transformcloud_benchmark
    ${PCL_LIBRARIES}
    glog::glog
    benchmark::benchmark
  )

  install(TARGETS registration_benchmark parallel_benchmark downsampling_benchmark transformcloud_benchmark
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()