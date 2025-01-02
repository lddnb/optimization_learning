/**
 * @ Author: Your name
 * @ Create Time: 1970-01-01 08:00:00
 * @ Modified by: lddnb
 * @ Modified time: 2025-01-02 11:47:35
 * @ Description:
 */

#pragma once

#include "optimization_learning/common.hpp"
#include "optimization_learning/registration_base.hpp"

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/registration_helper.hpp>

// 计算点云协方差
template <typename PointT>
std::vector<Eigen::Matrix3d> ComputeCovariancePSTL(const typename pcl::PointCloud<PointT>::Ptr& points_ptr, int num_neighbors)
{
  std::vector<Eigen::Matrix3d> covariances(points_ptr->size(), Eigen::Matrix3d::Zero());
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(points_ptr);
  std::vector<int> index(points_ptr->size());
  std::iota(index.begin(), index.end(), 0);

  std::for_each(std::execution::par, index.begin(), index.end(), [&](const int& i) {
    const PointT point = points_ptr->at(i);  // 获取当前点的索引
    std::vector<int> nn_indices(num_neighbors);
    std::vector<float> nn_distances(num_neighbors);
    const int n = kdtree.nearestKSearch(point, num_neighbors, nn_indices, nn_distances);
    if (n < 5 || nn_distances.back() > 1) {
      return;  // 跳过不满足条件的点
    }

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (size_t j = 0; j < n; ++j) {
      const auto& idx = nn_indices[j];
      const Eigen::Vector3d cur_p(points_ptr->at(idx).x, points_ptr->at(idx).y, points_ptr->at(idx).z);
      mean += cur_p;
      covariance += cur_p * cur_p.transpose();
    }
    mean = mean / n;
    covariance = covariance / n - mean * mean.transpose();
    covariances[i] = covariance;
  });

  return covariances;
}

template <typename PointT>
std::vector<Eigen::Matrix3d> ComputeCovarianceOMP(const typename pcl::PointCloud<PointT>::Ptr& points_ptr, int num_neighbors)
{
  std::vector<Eigen::Matrix3d> covariances(points_ptr->size(), Eigen::Matrix3d::Zero());
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(points_ptr);
  
  #pragma omp parallel for
  for (size_t i = 0; i < points_ptr->size(); ++i) {
    const PointT point = points_ptr->at(i);
    std::vector<int> nn_indices(num_neighbors);
    std::vector<float> nn_distances(num_neighbors);
    const int n = kdtree.nearestKSearch(point, num_neighbors, nn_indices, nn_distances);
    if (n < 5 || nn_distances.back() > 1) {
      continue; 
    }

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (size_t j = 0; j < n; ++j) {
      const auto& idx = nn_indices[j];
      const Eigen::Vector3d cur_p(points_ptr->at(idx).x, points_ptr->at(idx).y, points_ptr->at(idx).z);
      mean += cur_p;
      covariance += cur_p * cur_p.transpose();
    }
    mean = mean / n;
    covariance = covariance / n - mean * mean.transpose();
    covariances[i] = covariance;
  }

  return covariances;
}


template <typename PointT>
std::vector<Eigen::Matrix3d> ComputeCovarianceSEQ(const typename pcl::PointCloud<PointT>::Ptr& points_ptr, int num_neighbors)
{
  std::vector<Eigen::Matrix3d> covariances(points_ptr->size(), Eigen::Matrix3d::Zero());
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(points_ptr);
  
  for (size_t i = 0; i < points_ptr->size(); ++i) {
    const PointT point = points_ptr->at(i);
    std::vector<int> nn_indices(num_neighbors);
    std::vector<float> nn_distances(num_neighbors);
    const int n = kdtree.nearestKSearch(point, num_neighbors, nn_indices, nn_distances);
    if (n < 5 || nn_distances.back() > 1) {
      continue; 
    }

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (size_t j = 0; j < n; ++j) {
      const auto& idx = nn_indices[j];
      const Eigen::Vector3d cur_p(points_ptr->at(idx).x, points_ptr->at(idx).y, points_ptr->at(idx).z);
      mean += cur_p;
      covariance += cur_p * cur_p.transpose();
    }
    mean = mean / n;
    covariance = covariance / n - mean * mean.transpose();
    covariances[i] = covariance;
  }

  return covariances;
}

// Ceres

class CeresCostFunctorGICP
{
public:
  CeresCostFunctorGICP(
    const Eigen::Vector3d& curr_point,
    const Eigen::Vector3d& target_point,
    const Eigen::Matrix3d& information)
  : curr_point_(curr_point),
    target_point_(target_point),
    sqrt_information_(information.llt().matrixU())
  {
  }

  CeresCostFunctorGICP(
    const pcl::PointXYZI& curr_point,
    const pcl::PointXYZI& target_point,
    const Eigen::Matrix3d& information)
  : curr_point_(curr_point.x, curr_point.y, curr_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    sqrt_information_(information.llt().matrixU())
  {
  }

  template <typename T>
  bool operator()(const T* const se3, T* residual) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> R(se3);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(se3 + 4);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> weighted_error(residual);

    Eigen::Matrix<T, 3, 1> error = R * curr_point_.cast<T>() + t - target_point_.cast<T>();
    weighted_error = sqrt_information_.cast<T>() * error;
    return true;
  }
  
private:
  Eigen::Vector3d curr_point_;
  Eigen::Vector3d target_point_;
  Eigen::Matrix3d sqrt_information_;
};

// GTSAM SE3
class GtsamGICPFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GtsamGICPFactor(
    gtsam::Key key,
    const gtsam::Point3& source_point,
    const gtsam::Point3& target_point,
    const gtsam::Matrix3& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point),
    target_point_(target_point),
    sqrt_information_(information.llt().matrixU())
  {
  }

  GtsamGICPFactor(
    gtsam::Key key,
    const pcl::PointXYZI& source_point,
    const pcl::PointXYZI& target_point,
    const Eigen::Matrix3d& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    sqrt_information_(information.llt().matrixU())
  {
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Pose3& T,
    boost::optional<gtsam::Matrix&> H = boost::none) const override
  {
    gtsam::Matrix A = gtsam::Matrix::Zero(3, 6);
    gtsam::Point3 p_trans = T.transformFrom(source_point_, A);
    gtsam::Vector error = p_trans - target_point_;
    gtsam::Vector weighted_error = sqrt_information_ * error;

    if (H) {
      *H = sqrt_information_ * A;
    }
    return weighted_error;
  }

private:
  gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
  gtsam::Matrix3 sqrt_information_;
};

// GTSAM SO3 + R3
class GtsamGICPFactor2 : public gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>
{
public:
  GtsamGICPFactor2(
    gtsam::Key key1,
    gtsam::Key key2,
    const gtsam::Point3& source_point,
    const gtsam::Point3& target_point,
    const gtsam::Matrix3& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point),
    target_point_(target_point),
    sqrt_information_(information.llt().matrixU())
  {
  }

  GtsamGICPFactor2(
    gtsam::Key key1,
    gtsam::Key key2,
    const pcl::PointXYZI& source_point,
    const pcl::PointXYZI& target_point,
    const Eigen::Matrix3d& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    sqrt_information_(information.llt().matrixU())
  {
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Rot3& R,
    const gtsam::Point3& t,
    boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none) const override
  {
    gtsam::Vector error = R * source_point_ + t - target_point_;
    gtsam::Vector weighted_error = sqrt_information_ * error;
    
    if (H1) {  // 对旋转的雅克比
      // d(R*p)/dR = -R*（p^)
      *H1 = -sqrt_information_ * R.matrix() * gtsam::SO3::Hat(source_point_);
    }
    
    if (H2) {  // 对平移的雅克比
      // d(t)/dt = I
      *H2 = sqrt_information_;
    }
    
    return weighted_error;
  }

private:
  gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
  gtsam::Matrix3 sqrt_information_;
};

template <typename PointT>
void GICP_Ceres(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  Eigen::Quaterniond last_R = Eigen::Quaterniond(result_pose.rotation());
  Eigen::Vector3d last_t = result_pose.translation();
  std::vector<double> T = {last_R.x(), last_R.y(), last_R.z(), last_R.w(), last_t.x(), last_t.y(), last_t.z()};

  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);

  auto source_covariance = ComputeCovariancePSTL<PointT>(source_cloud_ptr, config.num_neighbors);
  auto target_covariance = ComputeCovariancePSTL<PointT>(target_cloud_ptr, config.num_neighbors);

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R.toRotationMatrix();
    T_opt.translation() = last_t;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<CeresCostFunctorGICP *> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, nn_indices, nn_distances) > 0;

      if (
          !valid || nn_distances[0] > config.max_correspondence_distance || source_covariance[idx].isZero() ||
          target_covariance[nn_indices.front()].isZero()) {
        return;
      }

      Eigen::Matrix3d cov = target_covariance[nn_indices.front()] + last_R.toRotationMatrix() * source_covariance[idx] * last_R.toRotationMatrix().transpose();

      cov += 1e-3 * Eigen::Matrix3d::Identity(); // 防止协方差矩阵奇异

      Eigen::Matrix3d cov_inv = cov.inverse();

      cost_functors[idx] =
        new CeresCostFunctorGICP(source_cloud_ptr->at(idx), target_cloud_ptr->at(nn_indices[0]), cov_inv);
    });

    ceres::Problem problem;
    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    problem.AddParameterBlock(T.data(), 7, se3);

    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctorGICP, 3, 7>(cost_functor);
      problem.AddResidualBlock(cost_function, nullptr, T.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = config.num_threads;
    options.minimizer_progress_to_stdout = config.verbose;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Map<Eigen::Quaterniond> R(T.data());
    Eigen::Map<Eigen::Vector3d> t(T.data() + 4);
    // todo：收敛阈值过小时会在两个点之间来回迭代，到达最大迭代次数后退出，原因未知

    if ((R.coeffs() - last_R.coeffs()).norm() < config.rotation_eps && 
        (t - last_t).norm() < config.translation_eps) {
      break;
    }
    last_R = R;
    last_t = t;
  }

  result_pose.translation() = last_t;
  result_pose.linear() = last_R.toRotationMatrix();
  num_iterations = iterations;
}

// GTSAM SE3
template <typename PointT>
void GICP_GTSAM_SE3(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  gtsam::Pose3 last_T_gtsam = gtsam::Pose3(gtsam::Rot3(result_pose.rotation()), gtsam::Point3(result_pose.translation()));
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);
  auto source_covariance = ComputeCovariancePSTL<PointT>(source_cloud_ptr, config.num_neighbors);
  auto target_covariance = ComputeCovariancePSTL<PointT>(target_cloud_ptr, config.num_neighbors);
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt(last_T_gtsam.matrix());
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<GtsamGICPFactor *> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索和构建因子
    std::for_each(
      std::execution::par,
      index.begin(),
      index.end(),
      [&](int idx) {
        
        // 近邻搜索
        std::vector<int> nn_indices(config.num_neighbors);
        std::vector<float> nn_distances(config.num_neighbors);
        bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, 
                                         nn_indices, nn_distances) > 0;
        
        if (
            !valid || nn_distances[0] > config.max_correspondence_distance || source_covariance[idx].isZero() ||
            target_covariance[nn_indices.front()].isZero()) {
          return;
        }

        Eigen::Matrix3d cov = target_covariance[nn_indices.front()] + last_T_gtsam.rotation().matrix() *
                                                                        source_covariance[idx] *
                                                                        last_T_gtsam.rotation().matrix().transpose();

        cov += 1e-3 * Eigen::Matrix3d::Identity(); // 防止协方差矩阵奇异
  
        Eigen::Matrix3d cov_inv = cov.inverse();

        // 构建因子
        cost_functors[idx] = new GtsamGICPFactor(
          key,
          source_cloud_ptr->at(idx), 
          target_cloud_ptr->at(nn_indices[0]), 
          cov_inv,
          noise_model);
      });

    // 串行添加因子
    gtsam::NonlinearFactorGraph graph;
    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      graph.add(*cost_functor);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key, last_T_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);
    optimizer.optimize();

    auto result = optimizer.values();
    gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key);

    if (
      (last_T_gtsam.rotation().toQuaternion().coeffs() -
       T_result.rotation().toQuaternion().coeffs()).norm() < config.rotation_eps &&
      (last_T_gtsam.translation() - T_result.translation()).norm() < config.translation_eps) {
      break;
    }
    last_T_gtsam = T_result;
  }
  num_iterations = iterations;
  
  result_pose = Eigen::Isometry3d(last_T_gtsam.matrix());
}

// GTSAM SO3 + R3
template <typename PointT>
void GICP_GTSAM_SO3_R3(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  gtsam::Rot3 last_R_gtsam = gtsam::Rot3(result_pose.rotation());
  gtsam::Point3 last_t_gtsam = gtsam::Point3(result_pose.translation());
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);
  auto source_covariance = ComputeCovariancePSTL<PointT>(source_cloud_ptr, config.num_neighbors);
  auto target_covariance = ComputeCovariancePSTL<PointT>(target_cloud_ptr, config.num_neighbors);
  const gtsam::Key key1 = gtsam::symbol_shorthand::X(0);
  const gtsam::Key key2 = gtsam::symbol_shorthand::X(1);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R_gtsam.matrix();
    T_opt.translation() = last_t_gtsam;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<GtsamGICPFactor2 *> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索和构建因子
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
        
        if (
            !valid || nn_distances[0] > config.max_correspondence_distance || source_covariance[idx].isZero() ||
            target_covariance[nn_indices.front()].isZero()) {
          return;
        }
  
        Eigen::Matrix3d cov = target_covariance[nn_indices.front()] + last_R_gtsam.matrix() * source_covariance[idx] * last_R_gtsam.matrix().transpose();
  
        cov += 1e-3 * Eigen::Matrix3d::Identity(); // 防止协方差矩阵奇异
  
        Eigen::Matrix3d cov_inv = cov.inverse();

        // 构建因子
        cost_functors[idx] = new GtsamGICPFactor2(
          key1, 
          key2,
          source_cloud_ptr->at(idx), 
          target_cloud_ptr->at(nn_indices[0]), 
          cov_inv,
          noise_model);
      });

    // 串行添加因子
    gtsam::NonlinearFactorGraph graph;
    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      graph.add(*cost_functor);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key1, last_R_gtsam);
    initial_estimate.insert(key2, last_t_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);
    optimizer.optimize();

    auto result = optimizer.values();
    gtsam::Rot3 R_result = result.at<gtsam::Rot3>(key1);
    gtsam::Point3 t_result = result.at<gtsam::Point3>(key2);

    if (
      (R_result.toQuaternion().coeffs() - last_R_gtsam.toQuaternion().coeffs()).norm() < config.rotation_eps &&
      (t_result - last_t_gtsam).norm() < config.translation_eps) {
      break;
    }
    last_R_gtsam = R_result;
    last_t_gtsam = t_result;
  }
  num_iterations = iterations;
  result_pose.translation() = last_t_gtsam;
  result_pose.linear() = last_R_gtsam.matrix();
}

// Gauss-Newton's method solve GICP.
template <typename PointT>
void GICP_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);
  auto source_covariance = ComputeCovariancePSTL<PointT>(source_cloud_ptr, config.num_neighbors);
  auto target_covariance = ComputeCovariancePSTL<PointT>(target_cloud_ptr, config.num_neighbors);
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

        if (
          !valid || nn_distances[0] > config.max_correspondence_distance || source_covariance[idx].isZero() ||
          target_covariance[nn_indices.front()].isZero()) {
          return;
        }
        Eigen::Matrix3d R = last_T.block<3, 3>(0, 0);
        Eigen::Matrix3d cov = target_covariance[nn_indices.front()] + R * source_covariance[idx] * R.transpose();

        cov += 1e-3 * Eigen::Matrix3d::Identity(); // 防止协方差矩阵奇异

        Eigen::Matrix3d cov_inv = cov.inverse();
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
        Eigen::Vector3d error = curr_point - target_point;

        Eigen::Matrix<double, 3, 6> Jacobian = Eigen::Matrix<double, 3, 6>::Zero();
        // 构建雅克比矩阵
        Jacobian.leftCols(3) = Eigen::Matrix3d::Identity();
        Jacobian.rightCols(3) = -R * Hat(source_point);
        Hs[idx] = Jacobian.transpose() * cov_inv * Jacobian;
        bs[idx] = -Jacobian.transpose() * cov_inv * error;
      });
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

template <typename PointT>
void GICP_PCL(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
  gicp.setInputSource(source_cloud_ptr);
  gicp.setInputTarget(target_cloud_ptr);
  gicp.setMaxCorrespondenceDistance(config.max_correspondence_distance);
  gicp.setTransformationEpsilon(config.translation_eps);
  gicp.setEuclideanFitnessEpsilon(config.translation_eps);
  gicp.setMaximumIterations(config.max_iterations);

  typename pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>);
  gicp.align(*aligned, result_pose.matrix().cast<float>());
  result_pose = gicp.getFinalTransformation().template cast<double>();
  num_iterations = gicp.nr_iterations_;
}

template <typename PointT>
void GICP_small_gicp(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
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
  small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP> registration;
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

template <typename PointT>
class GICPRegistration : public RegistrationBase<PointT>
{
public:
  GICPRegistration(const RegistrationConfig& config) : RegistrationBase<PointT>(config) {}

  void align(Eigen::Isometry3d& result_pose, int& num_iterations) override
  {
    result_pose = this->initial_transformation_;
    switch (this->config_.solve_type) {
      case RegistrationConfig::Ceres: {
        GICP_Ceres<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SE3: {
        GICP_GTSAM_SE3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SO3_R3: {
        GICP_GTSAM_SO3_R3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GN: {
        GICP_GN<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::PCL: {
        GICP_PCL<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::Koide: {
        GICP_small_gicp<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      default: {
        LOG(ERROR) << "Unknown registration solver method: " << this->config_.solve_type;
        break;
      }
    }
  }
};