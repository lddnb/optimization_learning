/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-19 15:04:38
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-23 18:57:23
 * @ Description:
 */

#pragma once

#include "common.hpp"

struct PointToPlaneICPConfig
{
double voxel_resolution = 1.0;
double downsampling_resolution = 0.25;
double max_correspondence_distance = 1.0;
double rotation_eps = 1e-3;  // 0.1 * M_PI / 180.0
double translation_eps = 1e-3;
int num_threads = 4;
int max_iterations = 30;
bool verbose = false;

int num_neighbors = 10;
};

// https://github.com/gaoxiang12/slam_in_autonomous_driving/blob/eb65a948353019a17c1d1ccb9ff8784bd25a6adf/src/common/math_utils.h#L112
template <typename S>
bool FitPlane(std::vector<Eigen::Matrix<S, 3, 1>>& data, Eigen::Matrix<S, 4, 1>& plane_coeffs, double eps = 1e-2) {
    if (data.size() < 3) {
        return false;
    }

    Eigen::MatrixXd A(data.size(), 4);
    for (int i = 0; i < data.size(); ++i) {
        A.row(i).head<3>() = data[i].transpose();
        A.row(i)[3] = 1.0;
    }

    Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
    plane_coeffs = svd.matrixV().col(3);

    // check error eps
    for (int i = 0; i < data.size(); ++i) {
        double err = plane_coeffs.template head<3>().dot(data[i]) + plane_coeffs[3];
        if (err * err > eps) {
            return false;
        }
    }

    return true;
}

class CeresCostFunctorP2Plane
{
public:
  CeresCostFunctorP2Plane(
    const Eigen::Vector3d& curr_point,
    const Eigen::Vector3d& target_point,
    const Eigen::Vector3d& normal)
  : curr_point_(curr_point),
    target_point_(target_point),
    normal_(normal)
  {
  }

  CeresCostFunctorP2Plane(
    const pcl::PointXYZI& curr_point,
    const pcl::PointXYZI& target_point,
    const Eigen::Vector3d& normal)
  : curr_point_(curr_point.x, curr_point.y, curr_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    normal_(normal)
  {
  }

  template <typename T>
  bool operator()(const T* const se3, T* residual) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> R(se3);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(se3 + 4);

    residual[0] =
      normal_.cast<T>().transpose() * (R * curr_point_.cast<T>() + t - target_point_.cast<T>());
    return true;
  }

private:
  Eigen::Vector3d curr_point_;
  Eigen::Vector3d target_point_;
  Eigen::Vector3d normal_;
};

// SE3
class GtsamIcpFactorP2Plane : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GtsamIcpFactorP2Plane(
    gtsam::Key key,
    const gtsam::Point3& source_point,
    const gtsam::Point3& target_point,
    const gtsam::Point3& normal,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point),
    target_point_(target_point),
    normal_(normal)
  {
  }

  GtsamIcpFactorP2Plane(
    gtsam::Key key,
    const pcl::PointXYZI& source_point,
    const pcl::PointXYZI& target_point,
    const Eigen::Vector3d& normal,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    normal_(normal)
  {
  }

  virtual gtsam::Vector evaluateError(const gtsam::Pose3& T, boost::optional<gtsam::Matrix&> H = boost::none) const override
  {
    gtsam::Matrix A = gtsam::Matrix::Zero(3, 6);

    gtsam::Point3 p_trans = T.transformFrom(source_point_, A);
    gtsam::Vector error = normal_.transpose() * (p_trans - target_point_);
    if (H) {
      *H = normal_.transpose() * A;

      // gtsam::Matrix J = gtsam::Matrix::Zero(1, 6);
      // J.leftCols(3) = -normal_.transpose() * T.rotation().matrix() * gtsam::SO3::Hat(source_point_);
      // J.rightCols(3) = normal_.transpose() * T.rotation().matrix();
      // *H = J;
    }
    return error;
  }

private:
  gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
  gtsam::Point3 normal_;
};


// SO3 + R3
class GtsamIcpFactorP2Plane2 : public gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>
{
public:
  GtsamIcpFactorP2Plane2(
    gtsam::Key key1,
    gtsam::Key key2,
    const gtsam::Point3& source_point,
    const gtsam::Point3& target_point,
    const gtsam::Point3& normal,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point),
    target_point_(target_point),
    normal_(normal)
  {
  }

  GtsamIcpFactorP2Plane2(
    gtsam::Key key1,
    gtsam::Key key2,
    const pcl::PointXYZI& source_point,
    const pcl::PointXYZI& target_point,
    const Eigen::Vector3d& normal,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    normal_(normal)
  {
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Rot3& R,
    const gtsam::Point3& t,
    boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none) const override
  {
    gtsam::Point3 p_trans = R * source_point_ + t;
    gtsam::Vector error = normal_.transpose() * (p_trans - target_point_);
    if (H1) {
      *H1 = -normal_.transpose() * R.matrix() * gtsam::SO3::Hat(source_point_);
    }
    if (H2) {
      *H2 = normal_.transpose();
    }
    return error;
  }

private:
  gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
  gtsam::Point3 normal_;
};


template <typename PointT>
void P2PlaneICP_GTSAM_SO3_R3(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const PointToPlaneICPConfig& config)
{
  gtsam::Rot3 last_R_gtsam = gtsam::Rot3(result_pose.rotation());
  gtsam::Point3 last_t_gtsam = gtsam::Point3(result_pose.translation());
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree = pcl::KdTreeFLANN<PointT>();
  kdtree.setInputCloud(target_cloud_ptr);
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
  const gtsam::Key key2 = gtsam::symbol_shorthand::X(1);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(1, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;
  
  for (int iterations = 0; iterations < config.max_iterations; ++iterations) {
    Eigen::Affine3d T_opt(Eigen::Translation3d(last_t_gtsam) * last_R_gtsam.matrix());
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt);

    gtsam::NonlinearFactorGraph graph;
    for (int i = 0; i < source_cloud_ptr->size(); ++i) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree.nearestKSearch(source_points_transformed->at(i), config.num_neighbors, nn_indices, nn_distances);

      std::vector<Eigen::Vector3d> plane_points;
      for (size_t i = 0; i < config.num_neighbors; ++i) {
        plane_points.emplace_back(
          target_cloud_ptr->at(nn_indices[i]).x,
          target_cloud_ptr->at(nn_indices[i]).y,
          target_cloud_ptr->at(nn_indices[i]).z);
      }
      Eigen::Matrix<double, 4, 1> plane_coeffs;
       if (nn_distances[0] > config.max_correspondence_distance || !FitPlane(plane_points, plane_coeffs)) {
        continue;
      }
      graph.emplace_shared<GtsamIcpFactorP2Plane2>(key, key2, source_cloud_ptr->at(i), target_cloud_ptr->at(nn_indices[0]), plane_coeffs.head<3>(), noise_model);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key, last_R_gtsam);
    initial_estimate.insert(key2, last_t_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);

    optimizer.optimize();

    auto result = optimizer.values();
    gtsam::Rot3 R_result = result.at<gtsam::Rot3>(key);
    gtsam::Point3 t_result = result.at<gtsam::Point3>(key2);

    if (
      (R_result.toQuaternion().coeffs() - last_R_gtsam.toQuaternion().coeffs()).norm() < config.rotation_eps &&
      (t_result - last_t_gtsam).norm() < config.translation_eps) {
      break;
    }
    last_R_gtsam = R_result;
    last_t_gtsam = t_result;
    num_iterations = iterations;
  }

  result_pose = Eigen::Affine3d(Eigen::Translation3d(last_t_gtsam) * last_R_gtsam.matrix());
}

// Gauss-Newton's method solve NICP.
template <typename PointT>
void P2PlaneICP_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const PointToPlaneICPConfig& config)
{
  bool has_converge_ = false;
  typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);
  typename pcl::KdTreeFLANN<PointT>::Ptr kdtree_flann_ptr_(new pcl::KdTreeFLANN<PointT>());
  kdtree_flann_ptr_->setInputCloud(target_cloud_ptr);

  Eigen::Matrix4d T = result_pose.matrix();

  // Gauss-Newton's method solve ICP.
  unsigned int i = 0;
  for (; i < config.max_iterations; ++i) {
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
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree_flann_ptr_->nearestKSearch(transformed_point, config.num_neighbors, nn_indices, nn_distances);

      std::vector<Eigen::Vector3d> plane_points;
      for (size_t i = 0; i < config.num_neighbors; ++i) {
        plane_points.emplace_back(
          target_cloud_ptr->at(nn_indices[i]).x,
          target_cloud_ptr->at(nn_indices[i]).y,
          target_cloud_ptr->at(nn_indices[i]).z);
      }
      Eigen::Matrix<double, 4, 1> plane_coeffs;
       if (nn_distances[0] > config.max_correspondence_distance || !FitPlane(plane_points, plane_coeffs)) {
        continue;
      }

      Eigen::Vector3d normal = plane_coeffs.head<3>();

      Eigen::Vector3d nearest_point = Eigen::Vector3d(
        target_cloud_ptr->at(nn_indices.front()).x,
        target_cloud_ptr->at(nn_indices.front()).y,
        target_cloud_ptr->at(nn_indices.front()).z);

      Eigen::Vector3d point_eigen(transformed_point.x, transformed_point.y, transformed_point.z);
      Eigen::Vector3d origin_point_eigen(origin_point.x, origin_point.y, origin_point.z);
      double error = normal.transpose() * (point_eigen - nearest_point);

      Eigen::Matrix<double, 1, 6> Jacobian = Eigen::Matrix<double, 1, 6>::Zero();
      // 构建雅克比矩阵
      Jacobian.leftCols(3) = normal.transpose();
      Jacobian.rightCols(3) = -normal.transpose() * T.block<3, 3>(0, 0) * Hat(origin_point_eigen);

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

    if (delta_x.norm() < config.translation_eps) {
      has_converge_ = true;
      break;
    }
  }
  num_iterations = i;

  result_pose = T;
}


template <typename PointT>
void P2PlaneICP_PCL(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const PointToPlaneICPConfig& config)
{
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr source_cloud_with_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_cloud_with_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> norm_est;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
  norm_est.setKSearch(config.num_neighbors);
  norm_est.setNumberOfThreads(config.num_threads);
  norm_est.setSearchMethod(tree);
  norm_est.setInputCloud(source_cloud_ptr);
  norm_est.compute(*normals);
  pcl::concatenateFields(*source_cloud_ptr, *normals, *source_cloud_with_normal);

  norm_est.setInputCloud(target_cloud_ptr);
  norm_est.compute(*normals);
  pcl::concatenateFields(*target_cloud_ptr, *normals, *target_cloud_with_normal);

  pcl::IterativeClosestPointWithNormals<pcl::PointXYZINormal, pcl::PointXYZINormal> nicp;
  nicp.setInputSource(source_cloud_with_normal);
  nicp.setInputTarget(target_cloud_with_normal);
  nicp.setMaxCorrespondenceDistance(config.max_correspondence_distance);
  nicp.setTransformationEpsilon(1e-5);
  nicp.setEuclideanFitnessEpsilon(1e-5);
  nicp.setMaximumIterations(config.max_iterations);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZINormal>);
  nicp.align(*aligned, result_pose.matrix().cast<float>());
  
  result_pose = nicp.getFinalTransformation().cast<double>();
}


// small_gicp point-to-plane ICP
template <typename PointT>
void P2PlaneICP_small_gicp(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose,
  int& num_iterations,
  const PointToPlaneICPConfig& config)
{
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
    small_gicp::KdTreeBuilderOMP(config.num_threads));
  auto source_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
    source,
    small_gicp::KdTreeBuilderOMP(config.num_threads));

  // Estimate point covariances
  estimate_covariances_omp(*target, *target_tree, config.num_neighbors, config.num_threads);
  estimate_covariances_omp(*source, *source_tree, config.num_neighbors, config.num_threads);

  // GICP + OMP-based parallel reduction
  small_gicp::Registration<small_gicp::PointToPlaneICPFactor, small_gicp::ParallelReductionOMP> registration;
  registration.reduction.num_threads = config.num_threads;
  registration.rejector.max_dist_sq = config.max_correspondence_distance * config.max_correspondence_distance;
  registration.criteria.rotation_eps = config.rotation_eps;
  registration.criteria.translation_eps = config.translation_eps;
  registration.optimizer.max_iterations = config.max_iterations;
  registration.optimizer.verbose = config.verbose;

  // Align point clouds
  Eigen::Isometry3d init_T_target_source(result_pose.matrix());
  auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

  result_pose = result.T_target_source;
  num_iterations = result.iterations;
}