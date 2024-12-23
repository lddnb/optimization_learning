cmake_minimum_required(VERSION 3.20)
project(optimization_learning)

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

add_subdirectory(thirdparty/small_gicp)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  thirdparty/small_gicp/include
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

install(TARGETS R_mean icp
  DESTINATION lib/${PROJECT_NAME}
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
  ament_add_gtest(test_icp test/test_icp.cpp)
  ament_target_dependencies(test_icp
    rclcpp
  )
  target_link_libraries(test_icp
    ${PCL_LIBRARIES}
    Ceres::ceres
    glog::glog
    gtsam
    small_gicp
  )
endif()

ament_package()