/**
 * @ Author: Your name
 * @ Create Time: 1970-01-01 08:00:00
 * @ Modified by: Your name
 * @ Modified time: 2024-12-30 22:50:53
 * @ Description:
 */

#pragma once

#include "common.hpp"

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include "small_gicp/factors/plane_icp_factor.hpp"
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/registration_helper.hpp>

struct GICPConfig
{
double downsampling_resolution = 0.25;
double max_correspondence_distance = 1.0;
double rotation_eps = 1e-3;  // 0.1 * M_PI / 180.0
double translation_eps = 1e-3;
int num_threads = 4;
int max_iterations = 30;
bool verbose = false;

int num_neighbors = 10;
};

// 计算点云协方差
template <typename PointT>
std::pair<Eigen::Vector3d, Eigen::Matrix3d> ComputeCovariance(const typename pcl::PointCloud<PointT>::Ptr& points_ptr)
{
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& point : points_ptr->points) {
    Eigen::Vector3d cur_p(point.x, point.y, point.z);
    mean += cur_p;
    covariance += cur_p * cur_p.transpose();
  }
  mean = mean / points_ptr->size();
  covariance = covariance - mean * mean.transpose();
  return {mean, covariance};
}

// Gauss-Newton's method solve GICP.
template <typename PointT>
void GICP_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const GICPConfig& config)
{
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);
  auto [source_mean, source_covariance] = ComputeCovariance<PointT>(source_cloud_ptr);
  auto [target_mean, target_covariance] = ComputeCovariance<PointT>(target_cloud_ptr);

  Eigen::Matrix4d last_T = result_pose.matrix();

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, last_T);
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
    std::vector<Eigen::Matrix<double, 6, 6>> Hs(source_points_transformed->size(), Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> bs(source_points_transformed->size(), Eigen::Matrix<double, 6, 1>::Zero());

    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    std::for_each(
      std::execution::par,
      index.begin(),
      index.end(),
      [&](int idx) {
        // 近邻搜索
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, 
                                         nn_indices, nn_distances) > 0;
        
        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

        Eigen::Matrix3d cov = target_covariance + last_T.block<3, 3>(0, 0) * source_covariance * last_T.block<3, 3>(0, 0).transpose();
        Eigen::Matrix3d sqrt_cov = cov.inverse().llt().matrixU();
        Eigen::Vector3d target_point = Eigen::Vector3d(
          target_cloud_ptr->at(nn_indices.front()).x,
          target_cloud_ptr->at(nn_indices.front()).y,
          target_cloud_ptr->at(nn_indices.front()).z);

        Eigen::Vector3d curr_point(source_points_transformed->at(idx).x, 
                                 source_points_transformed->at(idx).y, 
                                 source_points_transformed->at(idx).z);
        Eigen::Vector3d source_point(source_cloud_ptr->at(idx).x, 
                                   source_cloud_ptr->at(idx).y, 
                                   source_cloud_ptr->at(idx).z);
        Eigen::Vector3d error = sqrt_cov * (curr_point - target_point);

        Eigen::Matrix<double, 3, 6> Jacobian = Eigen::Matrix<double, 3, 6>::Zero();
        // 构建雅克比矩阵
        Jacobian.leftCols(3) = sqrt_cov;
        Jacobian.rightCols(3) = -sqrt_cov * last_T.block<3, 3>(0, 0) * Hat(source_point);
        Hs[idx] = Jacobian.transpose() * Jacobian;
        bs[idx] = -Jacobian.transpose() * error;
      });

    // 方案1：使用普通的 std::accumulate (串行执行)
    // auto result = std::accumulate(
    //   index.begin(),
    //   index.end(),
    //   H_b_type(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
    //   [&](const H_b_type& prev, int idx) -> H_b_type {
    //     const auto& J = Jacobians[idx];
    //     const double& e = errors[idx];
    //     if (e == 0.0) {
    //       return prev;
    //     }
    //     return H_b_type(
    //       prev.first + J.transpose() * J,
    //       prev.second - J.transpose() * e);
    //   });

    // 方案2：使用 std::transform_reduce (并行执行)
    auto result = std::transform_reduce(
      std::execution::par,
      index.begin(),
      index.end(),
      H_b_type(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
      // 规约操作
      [](const auto& a, const auto& b) {
        return std::make_pair(a.first + b.first, a.second + b.second);
      },
      // 转换操作
      [&Hs, &bs](const int& idx) {
        return H_b_type(Hs[idx], bs[idx]);
      }
    );

    H = result.first;
    b = result.second;

    if (H.determinant() == 0) {
      continue;
    }

    Eigen::Matrix<double, 6, 1> delta_x = H.inverse() * b;

    last_T.block<3, 1>(0, 3) = last_T.block<3, 1>(0, 3) + delta_x.head(3);
    last_T.block<3, 3>(0, 0) *= Exp(delta_x.tail(3)).matrix();

    if (delta_x.norm() < config.translation_eps) {
      break;
    }
  }
  num_iterations = iterations;

  result_pose = last_T;
}