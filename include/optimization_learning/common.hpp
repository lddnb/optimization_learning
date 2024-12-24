/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-23 17:58:13
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-23 18:06:29
 * @ Description:
 */

#pragma once

#include <execution>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include "small_gicp/factors/icp_factor.hpp"
#include "small_gicp/factors/plane_icp_factor.hpp"
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include <optimization_learning/so3_tool.hpp>

// bool next_iteration = false;
// void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
// {
//   if (event.getKeySym() == "space" && event.keyDown()) next_iteration = true;
// }


// pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
// viewer->setBackgroundColor(0, 0, 0);
// int v1(0);
// int v2(1);
// viewer->addPointCloud<pcl::PointXYZI>(source_points, "source_points");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_points");  // 红色
// viewer->addPointCloud<pcl::PointXYZI>(target_points, "target_points");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "target_points");  // 绿色
// viewer->addPointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "source_points_transformed");  // 蓝色

// viewer->spin();

// 逐步可视化
// viewer->registerKeyboardCallback (&keyboardEventOccurred, nullptr);
// while (!viewer->wasStopped ()) {
// viewer->spinOnce();
// if (next_iteration) {
//   // ...
//   viewer->updatePointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
// }
// next_iteration = false;
