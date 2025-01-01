/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-13 14:47:47
 * @ Modified by: Your name
 * @ Modified time: 2025-01-01 18:58:13
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
#include "small_gicp/factors/icp_factor.hpp"
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/registration/registration_helper.hpp>

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
template <typename PointT>
void P2PICP_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);

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

    // 并行执行近邻搜索和构建H、b
    std::for_each(
      std::execution::par,
      index.begin(),
      index.end(),
      [&](int idx) {
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, nn_indices, nn_distances) > 0;
        
        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

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
        Jacobian.leftCols(3) = Eigen::Matrix3d::Identity();
        Jacobian.rightCols(3) = -last_T.block<3, 3>(0, 0) * Hat(source_point);

        Hs[idx] = Jacobian.transpose() * Jacobian;
        bs[idx] = -Jacobian.transpose() * error;
      });

    // 并行规约求和
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

  result_pose = last_T;
  num_iterations = iterations;
}

template <typename PointT>
void P2PICP_Ceres(
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

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R.toRotationMatrix();
    T_opt.translation() = last_t;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<CeresCostFunctor*> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索和构建代价函数
    std::for_each(
      std::execution::par_unseq,
      index.begin(),
      index.end(),
      [&](int idx) {
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, nn_indices, nn_distances) > 0;

        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

        cost_functors[idx] = new CeresCostFunctor(
          source_cloud_ptr->at(idx),
          target_cloud_ptr->at(nn_indices[0]));
      });

    ceres::Problem problem;
    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    problem.AddParameterBlock(T.data(), 7, se3);

    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctor, 3, 7>(cost_functor);
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

template <typename PointT>
void P2PICP_GTSAM_SE3(
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

    std::vector<GtsamIcpFactor*> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索和构建因子
    std::for_each(
      std::execution::par,
      index.begin(),
      index.end(),
      [&](int idx) {
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, nn_indices, nn_distances) > 0;
        
        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

        cost_functors[idx] = new GtsamIcpFactor(
          key, 
          source_cloud_ptr->at(idx), 
          target_cloud_ptr->at(nn_indices[0]), 
          noise_model);
      });

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

    if ((last_T_gtsam.rotation().toQuaternion().coeffs() - T_result.rotation().toQuaternion().coeffs()).norm() < config.rotation_eps &&
        (last_T_gtsam.translation() - T_result.translation()).norm() < config.translation_eps) {
      break;
    }
    last_T_gtsam = T_result;
  }

  result_pose = Eigen::Isometry3d(last_T_gtsam.matrix());
  num_iterations = iterations;
}

template <typename PointT>
void P2PICP_GTSAM_SO3_R3(
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
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
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

    std::vector<GtsamIcpFactor2*> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索和构建因子
    std::for_each(
      std::execution::par,
      index.begin(),
      index.end(),
      [&](int idx) {
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);
        bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, nn_indices, nn_distances) > 0;
        
        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

        cost_functors[idx] = new GtsamIcpFactor2(
          key,
          key2,
          source_cloud_ptr->at(idx),
          target_cloud_ptr->at(nn_indices[0]),
          noise_model);
      });

    gtsam::NonlinearFactorGraph graph;
    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      graph.add(*cost_functor);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key, last_R_gtsam);
    initial_estimate.insert(key2, last_t_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);
    optimizer.optimize();

    auto result = optimizer.values();
    gtsam::Rot3 R_result = result.at<gtsam::Rot3>(key);
    gtsam::Point3 t_result = result.at<gtsam::Point3>(key2);

    if ((R_result.toQuaternion().coeffs() - last_R_gtsam.toQuaternion().coeffs()).norm() < config.rotation_eps &&
        (t_result - last_t_gtsam).norm() < config.translation_eps) {
      break;
    }
    last_R_gtsam = R_result;
    last_t_gtsam = t_result;
  }

  result_pose.translation() = last_t_gtsam;
  result_pose.linear() = last_R_gtsam.matrix();
  num_iterations = iterations;
}

template <typename PointT>
void P2PICP_PCL(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setInputSource(source_cloud_ptr);
  icp.setInputTarget(target_cloud_ptr);
  icp.setMaxCorrespondenceDistance(config.max_correspondence_distance);
  icp.setTransformationEpsilon(config.translation_eps);
  icp.setEuclideanFitnessEpsilon(config.translation_eps);
  icp.setMaximumIterations(config.max_iterations);

  typename pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>);
  icp.align(*aligned, result_pose.matrix().cast<float>());

  result_pose = Eigen::Isometry3d(icp.getFinalTransformation().template cast<double>());
  num_iterations = icp.nr_iterations_;
}

// small_gicp ICP
template <typename PointT>
void P2PICP_small_gicp(
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

  // GICP + OMP-based parallel reduction
  small_gicp::Registration<small_gicp::ICPFactor, small_gicp::ParallelReductionOMP> registration;
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
class ICPRegistration : public RegistrationBase<PointT>
{
public:
  ICPRegistration(const RegistrationConfig& config) : RegistrationBase<PointT>(config) {}

  void align(Eigen::Isometry3d& result_pose, int& num_iterations) override
  {
    result_pose = this->initial_transformation_;
    switch (this->config_.solve_type) {
      case RegistrationConfig::Ceres: {
        P2PICP_Ceres<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SE3: {
        P2PICP_GTSAM_SE3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SO3_R3: {
        P2PICP_GTSAM_SO3_R3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GN: {
        P2PICP_GN<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::PCL: {
        P2PICP_PCL<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::small_gicp: {
        P2PICP_small_gicp<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      default: {
        LOG(ERROR) << "Unknown registration solver method: " << this->config_.solve_type;
        break;
      }
    }
  }
};