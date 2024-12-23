/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-13 14:47:47
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-23 18:03:30
 * @ Description:
 */

#pragma once

#include "common.hpp"

class CeresCostFunctor
{
public:
  CeresCostFunctor(const Eigen::Vector3d& curr_point, const Eigen::Vector3d& target_point)
  : curr_point_(curr_point),
    target_point_(target_point)
  {
  }

  CeresCostFunctor(const pcl::PointXYZI& curr_point, const pcl::PointXYZI& target_point)
  : curr_point_(curr_point.x, curr_point.y, curr_point.z),
    target_point_(target_point.x, target_point.y, target_point.z)
  {
  }

  template <typename T>
  bool operator()(const T* const se3, T* residual) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> R(se3);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(se3 + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_eigen(residual);

    residual_eigen = R * curr_point_.cast<T>() + t - target_point_.cast<T>();
    return true;
  }

private:
  Eigen::Vector3d curr_point_;
  Eigen::Vector3d target_point_;
};

// Todo
// 和 R_mean 中相同的问题，用 ceres::EigenQuaternionManifold 残差先增再减
// 用 RightQuaternionManifold 和 j.topRows<3>().setIdentity() 收敛慢
class MyCossFunction : public ceres::SizedCostFunction<3, 4, 3>
{
public:
  MyCossFunction(const Eigen::Vector3d& curr_point, const Eigen::Vector3d& target_point)
  : curr_point_(curr_point),
    target_point_(target_point)
  {
  }

  virtual bool Evaluate(double const* const* se3, double* residuals, double** jacobians) const override
  {
    Eigen::Map<const Eigen::Quaterniond> R(se3[0]);
    Eigen::Map<const Eigen::Vector3d> t(se3[1]);
    Eigen::Map<Eigen::Vector3d> residual_eigen(residuals);

    residual_eigen = R * curr_point_ + t - target_point_;

    if (jacobians == nullptr) {
      return true;
    }

    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_eigen(jacobians[0]);
      jacobian_eigen.setZero();
      jacobian_eigen.leftCols(3) = -R.toRotationMatrix() * Hat(curr_point_);
    }

    if (jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_eigen(jacobians[1]);
      jacobian_eigen.setZero();
      jacobian_eigen = Eigen::Matrix<double, 3, 3>::Identity();
    }

    return true;
  }

private:
  Eigen::Vector3d curr_point_;
  Eigen::Vector3d target_point_;
};

// SE3
class GtsamIcpFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GtsamIcpFactor(
    gtsam::Key key,
    const gtsam::Point3& source_point,
    const gtsam::Point3& target_point,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point),
    target_point_(target_point)
  {
  }

  GtsamIcpFactor(
    gtsam::Key key,
    const pcl::PointXYZI& source_point,
    const pcl::PointXYZI& target_point,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z)
  {
  }

  virtual gtsam::Vector evaluateError(const gtsam::Pose3& T, boost::optional<gtsam::Matrix&> H = boost::none) const override
  {
    gtsam::Matrix A = gtsam::Matrix::Zero(3, 6);

    gtsam::Point3 p_trans = T.transformFrom(source_point_, A);
    gtsam::Vector error = p_trans - target_point_;
    if (H) {
      *H = A;

      // gtsam::Matrix J = gtsam::Matrix::Zero(3, 6);
      // J.leftCols(3) = -T.rotation().matrix() * gtsam::SO3::Hat(source_point_);
      // J.rightCols(3) = T.rotation().matrix();
      // *H = J;
    }
    return error;
  }

private:
  gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
};

// SO3 + R3
class GtsamIcpFactor2 : public gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>
{
public:
  GtsamIcpFactor2(
    gtsam::Key key1,
    gtsam::Key key2,
    const gtsam::Point3& source_point,
    const gtsam::Point3& target_point,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point),
    target_point_(target_point)
  {
  }

  GtsamIcpFactor2(
    gtsam::Key key1,
    gtsam::Key key2,
    const pcl::PointXYZI& source_point,
    const pcl::PointXYZI& target_point,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z)
  {
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Rot3& R,
    const gtsam::Point3& t,
    boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none) const override
  {
    gtsam::Point3 p_trans = R * source_point_ + t;
    gtsam::Vector error = p_trans - target_point_;
    if (H1) {
      *H1 = -R.matrix() * gtsam::SO3::Hat(source_point_);
    }
    if (H2) {
      *H2 = gtsam::I_3x3;
    }
    return error;
  }

private:
  gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
};

// Gauss-Newton's method solve ICP.
// https://github.com/zm0612/optimized_ICP/blob/be8651addd630c472418cf530a53623946906831/optimized_ICP_GN.cpp#L17
template <typename PointT>
bool MatchP2P(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const Eigen::Matrix4d& predict_pose,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose)
{
  bool has_converge_ = false;
  int max_iterations_ = 30;
  double max_correspond_distance_ = 0.5;
  double transformation_epsilon_ = 1e-5;

  typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);
  typename pcl::KdTreeFLANN<PointT>::Ptr kdtree_flann_ptr_(new pcl::KdTreeFLANN<PointT>());
  kdtree_flann_ptr_->setInputCloud(target_cloud_ptr);

  Eigen::Matrix4d T = predict_pose;

  // Gauss-Newton's method solve ICP.
  unsigned int i = 0;
  for (; i < max_iterations_; ++i) {
    pcl::transformPointCloud(*source_cloud_ptr, *transformed_cloud, T);
    Eigen::Matrix<double, 6, 6> Hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> B = Eigen::Matrix<double, 6, 1>::Zero();

    for (unsigned int j = 0; j < transformed_cloud->size(); ++j) {
      const PointT& origin_point = source_cloud_ptr->points[j];

      // 删除距离为无穷点
      if (!pcl::isFinite(origin_point)) {
        continue;
      }

      const PointT& transformed_point = transformed_cloud->at(j);
      std::vector<float> resultant_distances;
      std::vector<int> indices;
      // 在目标点云中搜索距离当前点最近的一个点
      kdtree_flann_ptr_->nearestKSearch(transformed_point, 1, indices, resultant_distances);

      // 舍弃那些最近点,但是距离大于最大对应点对距离
      if (resultant_distances.front() > max_correspond_distance_) {
        continue;
      }

      Eigen::Vector3d nearest_point = Eigen::Vector3d(
        target_cloud_ptr->at(indices.front()).x,
        target_cloud_ptr->at(indices.front()).y,
        target_cloud_ptr->at(indices.front()).z);

      Eigen::Vector3d point_eigen(transformed_point.x, transformed_point.y, transformed_point.z);
      Eigen::Vector3d origin_point_eigen(origin_point.x, origin_point.y, origin_point.z);
      Eigen::Vector3d error = point_eigen - nearest_point;

      Eigen::Matrix<double, 3, 6> Jacobian = Eigen::Matrix<double, 3, 6>::Zero();
      // 构建雅克比矩阵
      Jacobian.leftCols(3) = Eigen::Matrix3d::Identity();
      Jacobian.rightCols(3) = -T.block<3, 3>(0, 0) * Hat(origin_point_eigen);

      // 构建海森矩阵
      Hessian += Jacobian.transpose() * Jacobian;
      B += -Jacobian.transpose() * error;
    }

    if (Hessian.determinant() == 0) {
      continue;
    }

    Eigen::Matrix<double, 6, 1> delta_x = Hessian.inverse() * B;

    T.block<3, 1>(0, 3) = T.block<3, 1>(0, 3) + delta_x.head(3);
    T.block<3, 3>(0, 0) *= Exp(delta_x.tail(3)).matrix();

    if (delta_x.norm() < transformation_epsilon_) {
      has_converge_ = true;
      break;
    }

    // debug
    // LOG(INFO) << "i= " << i << "  norm delta x= " << delta_x.norm();
  }
  LOG(INFO) << "iterations: " << i;

  result_pose = T;

  return true;
}

// small_gicp ICP
template <typename PointT>
void ICP_small_gicp(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations)
{
  double voxel_resolution = 1.0;             ///< Voxel resolution for VGICP
  double downsampling_resolution = 0.25;     ///< Downsample resolution (this will be used only in the Eigen-based interface)
  double max_correspondence_distance = 1.0;  ///< Maximum correspondence distance
  double rotation_eps = 0.1 * M_PI / 180.0;  ///< Rotation tolerance for convergence check [rad]
  double translation_eps = 1e-3;             ///< Translation tolerance for convergence check
  int num_threads = 4;                       ///< Number of threads
  int max_iterations = 20;                   ///< Maximum number of iterations
  bool verbose = false;                      ///< Verbose mode

  int num_neighbors = 10;

  std::vector<Eigen::Vector3d> source_eigen(source_cloud_ptr->size());
  std::vector<Eigen::Vector3d> target_eigen(target_cloud_ptr->size());
  std::transform(
    std::execution::par,
    source_cloud_ptr->begin(),
    source_cloud_ptr->end(),
    source_eigen.begin(),
    [](const pcl::PointXYZI& point) { return Eigen::Vector3d(point.x, point.y, point.z); });
  std::transform(
    std::execution::par,
    target_cloud_ptr->begin(),
    target_cloud_ptr->end(),
    target_eigen.begin(),
    [](const pcl::PointXYZI& point) { return Eigen::Vector3d(point.x, point.y, point.z); });

  auto target = std::make_shared<small_gicp::PointCloud>(target_eigen);
  auto source = std::make_shared<small_gicp::PointCloud>(source_eigen);

  // Create KdTree
  auto target_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
    target,
    small_gicp::KdTreeBuilderOMP(num_threads));
  auto source_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
    source,
    small_gicp::KdTreeBuilderOMP(num_threads));

  // Estimate point covariances
  estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
  estimate_covariances_omp(*source, *source_tree, num_neighbors, num_threads);

  // GICP + OMP-based parallel reduction
  small_gicp::Registration<small_gicp::ICPFactor, small_gicp::ParallelReductionOMP> registration;
  registration.reduction.num_threads = num_threads;
  registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;
  registration.criteria.rotation_eps = rotation_eps;
  registration.criteria.translation_eps = translation_eps;
  registration.optimizer.max_iterations = max_iterations;
  registration.optimizer.verbose = verbose;

  // Align point clouds
  Eigen::Isometry3d init_T_target_source(result_pose.matrix());
  auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

  result_pose = result.T_target_source;
  num_iterations = result.iterations;
}