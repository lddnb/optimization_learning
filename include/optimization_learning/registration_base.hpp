/**
 * @ Author: Your name
 * @ Create Time: 1970-01-01 08:00:00
 * @ Modified by: Your name
 * @ Modified time: 2025-01-01 17:27:50
 * @ Description:
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

struct RegistrationConfig {
  enum RegistrationType {
    ICP = 0,
    NICP,
    GICP,
    NDT,
  };
  enum SolveType {
    Ceres = 0,
    GTSAM_SE3,
    GTSAM_SO3_R3,
    GN,
    PCL,
    small_gicp,
    OMP
  };
  RegistrationType registration_type = ICP;
  SolveType solve_type = Ceres;
  // ndt
  double resolution = 1;
  double residual_outlier_threshold = 20;
  int num_residual_per_point = 7;
  // common
  double max_correspondence_distance = 1.0;
  double rotation_eps = 1e-3;
  double translation_eps = 1e-3;
  int num_threads = 4;
  int max_iterations = 30;
  bool verbose = false;
  int num_neighbors = 10;
};

template <typename PointT>
class RegistrationBase
{
public:
  using Ptr = std::shared_ptr<RegistrationBase<PointT>>;

  RegistrationBase() = default;
  RegistrationBase(const RegistrationConfig& config) : config_(config) {}
  virtual ~RegistrationBase() = default;

  // 设置输入点云
  virtual void setInputSource(const typename pcl::PointCloud<PointT>::Ptr& source_cloud)
  {
    source_cloud_ = source_cloud;
  }

  virtual void setInputTarget(const typename pcl::PointCloud<PointT>::Ptr& target_cloud)
  {
    target_cloud_ = target_cloud;
  }

  // 设置初始变换矩阵
  virtual void setInitialTransformation(const Eigen::Isometry3d& initial_transformation)
  {
    initial_transformation_ = initial_transformation;
  }

  // 执行配准
  virtual void align(Eigen::Isometry3d& result_pose, int& num_iterations) = 0;

  virtual RegistrationConfig& config() { return config_; }

protected:
  typename pcl::PointCloud<PointT>::Ptr source_cloud_;
  typename pcl::PointCloud<PointT>::Ptr target_cloud_;
  Eigen::Isometry3d initial_transformation_;
  RegistrationConfig config_;
};