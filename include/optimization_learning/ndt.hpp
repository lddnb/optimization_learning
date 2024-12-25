/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-25 10:17:10
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-25 18:09:50
 * @ Description:
 */

#pragma once

#include "common.hpp"
#include "pclomp/ndt_omp.h"

struct EigenVec3iHash {
  std::size_t operator()(const Eigen::Vector3i& v) const
  {
    return ((std::hash<int>()(v.x()) ^ (std::hash<int>()(v.y()) << 1)) >> 1) ^
           (std::hash<int>()(v.z()) << 1);
  }
};

struct NDTConfig {
  double resolution = 1;
  double max_correspondence_distance = 1.0;
  double step_size = 0.1;
  double epsilon = 1e-3;
  int max_iterations = 30;
  double residual_outlier_threshold = 20;
  int num_residual_per_point = 7;
};

// 体素内点的统计信息
struct VoxelData {
  bool valid = false;
  std::vector<int> indices;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
};

// 体素网格
class VoxelGrid
{
public:
  explicit VoxelGrid(double resolution) : resolution_(resolution) {}

  void setCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
  {
    voxels_.clear();
    min_bound_ = Eigen::Vector3d(
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max());
    max_bound_ = Eigen::Vector3d(
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest());

    for(size_t idx = 0; idx < cloud->size(); ++idx) {
      Eigen::Vector3d p(cloud->at(idx).x, cloud->at(idx).y, cloud->at(idx).z);
      min_bound_ = min_bound_.cwiseMin(p);
      max_bound_ = max_bound_.cwiseMax(p);

      auto voxel_idx = getVoxelIndex(p);
      auto& voxel = voxels_[voxel_idx];
      voxel.indices.emplace_back(idx);
      voxel.mean += p;
      voxel.covariance += p * p.transpose();
    }

    // 计算体素数量
    grid_size_ = ((max_bound_ - min_bound_) / resolution_).array().ceil().cast<int>();

    std::for_each(std::execution::par_unseq, voxels_.begin(), voxels_.end(), [&](auto& pair) {
      auto& voxel = pair.second;
      int num_points = voxel.indices.size();
      if (num_points < 5) {  // 至少需要5个点才能计算协方差
        return;
      }
      voxel.mean /= num_points;
      voxel.covariance = voxel.covariance / num_points - voxel.mean * voxel.mean.transpose();

      // NDT (eq 6.11)[Magnusson 2009] 原文中的处理方式
      // 计算特征值和特征向量，特征值是按照升序排列
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(voxel.covariance);
      Eigen::Matrix3d eigen_val = eigensolver.eigenvalues().asDiagonal();
      Eigen::Matrix3d eigen_vec = eigensolver.eigenvectors();

      if (
        eigen_val(0, 0) < -Eigen::NumTraits<double>::dummy_precision() ||
        eigen_val(1, 1) < -Eigen::NumTraits<double>::dummy_precision() || eigen_val(2, 2) <= 0) {
        voxel.indices.clear();
        return;
      }
      double min_covar_eigvalue = 0.01 * eigen_val(2, 2);
      if (eigen_val(0, 0) < min_covar_eigvalue) {
        eigen_val(0, 0) = min_covar_eigvalue;

        if (eigen_val(1, 1) < min_covar_eigvalue) {
          eigen_val(1, 1) = min_covar_eigvalue;
        }

        voxel.covariance = eigen_vec * eigen_val * eigen_vec.inverse();
      }
      // 计算信息矩阵
      voxel.information = voxel.covariance.inverse();

      // SVD 分解的方式处理
      // 奇异值按降序排列
      // Eigen::JacobiSVD<Eigen::Matrix3d> svd(voxel.covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
      // Eigen::Matrix3d U = svd.matrixU();
      // Eigen::Matrix3d V = svd.matrixV();
      // Eigen::Vector3d S = svd.singularValues();
      // if (S[1] < S[0] * 1e-2) {
      //   S[1] = S[0] * 1e-2;
      // }
      // if (S[2] < S[0] * 1e-2) {
      //   S[2] = S[0] * 1e-2;
      // }
      // Eigen::Matrix3d S_inv = Eigen::Vector3d(1.0 / S[0], 1.0 / S[1], 1.0 / S[2]).asDiagonal();
      // voxel.information = V * S_inv * U.transpose();

      voxel.valid = true;
    });
  }

  VoxelData* getVoxel(const Eigen::Vector3d& p)
  {
    auto iter = voxels_.find(getVoxelIndex(p));
    return iter != voxels_.end() && iter->second.valid ? &iter->second : nullptr;
  }

  VoxelData* getVoxel(const Eigen::Vector3i& voxel_idx)
  {
    auto iter = voxels_.find(voxel_idx);
    return iter != voxels_.end() && iter->second.valid ? &iter->second : nullptr;
  }

  Eigen::Vector3i getVoxelIndex(const Eigen::Vector3d& p) const
  {
    return ((p - min_bound_) / resolution_).array().floor().cast<int>();
  }
private:
  double resolution_;
  Eigen::Vector3d min_bound_, max_bound_;
  Eigen::Vector3i grid_size_;
  std::unordered_map<Eigen::Vector3i, VoxelData, EigenVec3iHash> voxels_;
};

template <typename PointT>
void NDT_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const NDTConfig& config)
{
  // 构建目标点云的NDT体素
  VoxelGrid target_grid(config.resolution);
  target_grid.setCloud(target_cloud_ptr);

  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  Eigen::Matrix4d T = result_pose.matrix();

  std::vector<Eigen::Vector3i> nearby_grids_indices;
  if (config.num_residual_per_point == 7) {
    nearby_grids_indices = {
      Eigen::Vector3i(0, 0, 0),
      Eigen::Vector3i(0, 0, 1),
      Eigen::Vector3i(0, 1, 0),
      Eigen::Vector3i(1, 0, 0),
      Eigen::Vector3i(0, 0, -1),
      Eigen::Vector3i(0, -1, 0),
      Eigen::Vector3i(-1, 0, 0),
    };
  } else if (config.num_residual_per_point == 1) {
    nearby_grids_indices = {
      Eigen::Vector3i(0, 0, 0),
    };
  } else {
    LOG(ERROR) << "Invalid num_residual_per_point: " << config.num_residual_per_point;
    return;
  }

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T);

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

    std::vector<Eigen::Matrix<double, 6, 6>> Hs(source_points_transformed->size(), Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> bs(source_points_transformed->size(), Eigen::Matrix<double, 6, 1>::Zero());

    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    std::atomic<int> num_valid_points = 0;

    std::for_each(std::execution::par, index.begin(), index.end(), [&](int idx) {
      Eigen::Vector3d p_trans(
        source_points_transformed->at(idx).x,
        source_points_transformed->at(idx).y,
        source_points_transformed->at(idx).z);
      Eigen::Vector3i key = target_grid.getVoxelIndex(p_trans);
      Eigen::Matrix<double, 6, 6> H_sum = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> b_sum = Eigen::Matrix<double, 6, 1>::Zero();
      for (size_t i = 0; i < nearby_grids_indices.size(); ++i) {
        Eigen::Vector3i voxel_idx = key + nearby_grids_indices[i];
        auto* voxel = target_grid.getVoxel(voxel_idx);
        if (!voxel) continue;
        Eigen::Vector3d d = p_trans - voxel->mean;
        double cost = d.transpose() * voxel->information * d;
        if (std::abs(cost) > config.residual_outlier_threshold) return;

        num_valid_points.fetch_add(1);

        // 计算雅可比矩阵
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J.block<3, 3>(0, 3) = -T.block<3, 3>(0, 0) * Hat(p_trans);

        // 计算海森矩阵和梯度
        Eigen::Matrix3d W = voxel->information;
        H_sum += J.transpose() * W * J;
        b_sum += -J.transpose() * W * d;
      }
      Hs[idx] = H_sum;
      bs[idx] = b_sum;
    });

    // LOG(INFO) << "num_valid_points: " << num_valid_points.load();

    if (num_valid_points < 5) {
      continue;
    }

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

    // 求解增量
    Eigen::Matrix<double, 6, 1> delta = H.inverse() * b;

    // 更新位姿
    T.block<3, 1>(0, 3) += delta.head<3>();
    T.block<3, 3>(0, 0) *= Exp(delta.tail<3>()).matrix();

    // 收敛判断
    if (delta.norm() < config.epsilon) {
      break;
    }
  }

  result_pose = T;
  num_iterations = iterations;
}

template <typename PointT>
void NDT_PCL(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const NDTConfig& config)
{
  typename pcl::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pcl::NormalDistributionsTransform<PointT, PointT>());
  ndt->setResolution(config.resolution);
  ndt->setInputTarget(target_cloud_ptr);
  ndt->setInputSource(source_cloud_ptr);
  ndt->setMaximumIterations(config.max_iterations);
  ndt->setTransformationEpsilon(config.epsilon);
  ndt->setEuclideanFitnessEpsilon(config.epsilon);

  pcl::PointCloud<PointT> output_cloud;
  ndt->align(output_cloud, result_pose.matrix().cast<float>());

  result_pose = ndt->getFinalTransformation().template cast<double>();
  num_iterations = ndt->getFinalNumIteration();
}

template <typename PointT>
void NDT_OMP(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const NDTConfig& config)
{
  typename pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<PointT, PointT>());
  ndt_omp->setResolution(config.resolution);
  ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  ndt_omp->setInputTarget(target_cloud_ptr);
  ndt_omp->setInputSource(source_cloud_ptr);
  ndt_omp->setMaximumIterations(config.max_iterations);
  ndt_omp->setTransformationEpsilon(config.epsilon);
  ndt_omp->setEuclideanFitnessEpsilon(config.epsilon);

  pcl::PointCloud<PointT> output_cloud;
  ndt_omp->align(output_cloud, result_pose.matrix().cast<float>());

  result_pose = ndt_omp->getFinalTransformation().template cast<double>();
  num_iterations = ndt_omp->getFinalNumIteration();

  // // 创建输出点云的智能指针
  // typename pcl::PointCloud<PointT>::ConstPtr output_cloud_ptr(new pcl::PointCloud<PointT>(output_cloud));
  // typename pcl::PointCloud<PointT>::ConstPtr source_cloud_const_ptr = source_cloud_ptr;
  // typename pcl::PointCloud<PointT>::ConstPtr target_cloud_const_ptr = target_cloud_ptr;

  // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  // viewer->setBackgroundColor(0, 0, 0);
  
  // // 使用ConstPtr类型添加点云
  // // viewer->addPointCloud<PointT>(source_cloud_const_ptr, "source_points");
  // // viewer->setPointCloudRenderingProperties(
  // //   pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_points");  // 红色
    
  // viewer->addPointCloud<PointT>(target_cloud_const_ptr, "target_points");
  // viewer->setPointCloudRenderingProperties(
  //   pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "target_points");  // 绿色
    
  // viewer->addPointCloud<PointT>(output_cloud_ptr, "source_points_transformed");
  // viewer->setPointCloudRenderingProperties(
  //   pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "source_points_transformed");  // 蓝色

  // viewer->spin();
}