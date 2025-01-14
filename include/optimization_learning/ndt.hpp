/**
 * @file ndt.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include "optimization_learning/common.hpp"
#include "optimization_learning/registration_base.hpp"
#include "pclomp/ndt_omp.h"

struct EigenVec3iHash {
  std::size_t operator()(const Eigen::Vector3i& v) const
  {
    return ((std::hash<int>()(v.x()) ^ (std::hash<int>()(v.y()) << 1)) >> 1) ^
           (std::hash<int>()(v.z()) << 1);
  }
};

// 体素内点的统计信息
struct VoxelData {
  bool valid = false;
  std::vector<int> indices;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
};

inline std::vector<Eigen::Vector3i> getNearbyGridsIndices(int num_residual_per_point)
{
  if (num_residual_per_point == 7) {
    return {
      Eigen::Vector3i(0, 0, 0),
      Eigen::Vector3i(-1, 0, 0),
      Eigen::Vector3i(1, 0, 0),
      Eigen::Vector3i(0, -1, 0),
      Eigen::Vector3i(0, 1, 0),
      Eigen::Vector3i(0, 0, -1),
      Eigen::Vector3i(0, 0, 1),
    };
  } else if (num_residual_per_point == 1) {
    return {
      Eigen::Vector3i(0, 0, 0),
    };
  } else {
    return {};
  }
}

// 体素网格
class VoxelGrid
{
public:
  explicit VoxelGrid(double resolution) : resolution_(resolution) {}

  template <typename PointT>
  void setCloud(const pcl::PointCloud<PointT>& cloud)
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

    for(size_t idx = 0; idx < cloud.size(); ++idx) {
      Eigen::Vector3d p(cloud.at(idx).x, cloud.at(idx).y, cloud.at(idx).z);
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
      if (num_points < 3) {
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

  // Todo: 为什么不能减去 min_bound_？
  Eigen::Vector3i getVoxelIndex(const Eigen::Vector3d& p) const
  {
    return ((p) / resolution_).array().floor().cast<int>();
  }
private:
  double resolution_;
  Eigen::Vector3d min_bound_, max_bound_;
  Eigen::Vector3i grid_size_;
  std::unordered_map<Eigen::Vector3i, VoxelData, EigenVec3iHash> voxels_;
};

class CeresCostFunctorNDT
{
public:
  CeresCostFunctorNDT(
    const Eigen::Vector3d& curr_point,
    const Eigen::Vector3d& voxel_mean,
    const Eigen::Matrix3d& information)
  : curr_point_(curr_point),
    voxel_mean_(voxel_mean),
    sqrt_information_(information.llt().matrixU())
  {
  }

  template <typename PointT>
  CeresCostFunctorNDT(
    const PointT& curr_point,
    const Eigen::Vector3d& voxel_mean,
    const Eigen::Matrix3d& information)
  : curr_point_(curr_point.x, curr_point.y, curr_point.z),
    voxel_mean_(voxel_mean),
    sqrt_information_(information.llt().matrixU())
  {
  }

  template <typename T>
  bool operator()(const T* const se3, T* residuals) const {
    Eigen::Map<const Eigen::Quaternion<T>> q(se3);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(se3 + 4);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> weighted_error(residuals);
    
    // 计算残差向量
    Eigen::Matrix<T, 3, 1> error = q * curr_point_.cast<T>() + t - voxel_mean_.cast<T>();
    
    // 应用信息矩阵的平方根
    weighted_error = sqrt_information_.cast<T>() * error;
    
    return true;
  }

private:
  Eigen::Vector3d curr_point_;
  Eigen::Vector3d voxel_mean_;
  Eigen::Matrix3d sqrt_information_;  // 信息矩阵的平方根
};

// SE3
class GtsamNDTFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GtsamNDTFactor(
    gtsam::Key key,
    const gtsam::Point3& source_point,
    const gtsam::Point3& voxel_mean,
    const gtsam::Matrix3& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point),
    voxel_mean_(voxel_mean),
    sqrt_information_(information.llt().matrixU())
  {
  }

  template <typename PointT>
  GtsamNDTFactor(
    gtsam::Key key,
    const PointT& source_point,
    const Eigen::Vector3d& voxel_mean,
    const Eigen::Matrix3d& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point.x, source_point.y, source_point.z),
    voxel_mean_(voxel_mean),
    sqrt_information_(information.llt().matrixU())
  {
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Pose3& T,
    boost::optional<gtsam::Matrix&> H = boost::none) const override
  {
    gtsam::Matrix A = gtsam::Matrix::Zero(3, 6);
    gtsam::Point3 p_trans = T.transformFrom(source_point_, A);
    gtsam::Vector error = p_trans - voxel_mean_;
    gtsam::Vector weighted_error = sqrt_information_ * error;

    if (H) {
      *H = sqrt_information_ * A;
    }
    return weighted_error;
  }


private:
  gtsam::Point3 source_point_;
  gtsam::Point3 voxel_mean_;
  gtsam::Matrix3 sqrt_information_;
};

// SO3 + R3
class GtsamNDTFactor2 : public gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>
{
public:
  GtsamNDTFactor2(
    gtsam::Key key1,
    gtsam::Key key2,
    const gtsam::Point3& source_point,
    const gtsam::Point3& voxel_mean,
    const gtsam::Matrix3& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point),
    voxel_mean_(voxel_mean),
    sqrt_information_(information.llt().matrixU())
  {
  }

  template <typename PointT>
  GtsamNDTFactor2(
    gtsam::Key key1,
    gtsam::Key key2,
    const PointT& source_point,
    const Eigen::Vector3d& voxel_mean,
    const Eigen::Matrix3d& information,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point.x, source_point.y, source_point.z),
    voxel_mean_(voxel_mean),
    sqrt_information_(information.llt().matrixU())
  {
  }

  virtual gtsam::Vector evaluateError(
    const gtsam::Rot3& R,
    const gtsam::Point3& t,
    boost::optional<gtsam::Matrix&> H1 = boost::none,
    boost::optional<gtsam::Matrix&> H2 = boost::none) const override
  {
    gtsam::Vector error = R * source_point_ + t - voxel_mean_;
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
  gtsam::Point3 voxel_mean_;
  gtsam::Matrix3 sqrt_information_;
};

template <typename PointT>
void NDT_Ceres(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  // 构建目标点云的NDT体素
  VoxelGrid target_grid(config.resolution);
  target_grid.setCloud(*target_cloud_ptr);

  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  Eigen::Quaterniond last_R = Eigen::Quaterniond(result_pose.rotation());
  Eigen::Vector3d last_t = result_pose.translation();
  std::vector<double> T = {last_R.x(), last_R.y(), last_R.z(), last_R.w(), last_t.x(), last_t.y(), last_t.z()};

  std::vector<Eigen::Vector3i> nearby_grids_indices = getNearbyGridsIndices(config.num_residual_per_point);
  if (nearby_grids_indices.empty()) {
    LOG(ERROR) << "Invalid num_residual_per_point: " << config.num_residual_per_point;
    return;
  }

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R.toRotationMatrix();
    T_opt.translation() = last_t;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());
    
    std::vector<CeresCostFunctorNDT *> cost_functors(source_points_transformed->size() * nearby_grids_indices.size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
      Eigen::Vector3d p_trans(
        source_points_transformed->at(idx).x,
        source_points_transformed->at(idx).y,
        source_points_transformed->at(idx).z);
      Eigen::Vector3i key = target_grid.getVoxelIndex(p_trans);
      for (int i = 0; i < nearby_grids_indices.size(); ++i) {
        Eigen::Vector3i voxel_idx = key + nearby_grids_indices[i];
        auto* voxel = target_grid.getVoxel(voxel_idx);
        if (!voxel) continue;
        Eigen::Vector3d d = p_trans - voxel->mean;
        double cost = d.transpose() * voxel->information * d;
        if (std::abs(cost) > config.residual_outlier_threshold) continue;
        cost_functors[idx * nearby_grids_indices.size() + i] = new CeresCostFunctorNDT(source_cloud_ptr->at(idx), voxel->mean, voxel->information);
      }
    });

    ceres::Problem problem;
    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    problem.AddParameterBlock(T.data(), 7, se3);

    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctorNDT, 3, 7>(cost_functor);
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

    if ((R.coeffs() - last_R.coeffs()).norm() < config.translation_eps && 
        (t - last_t).norm() < config.rotation_eps) {
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
void NDT_GTSAM_SE3(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  // 构建目标点云的NDT体素
  VoxelGrid target_grid(config.resolution);
  target_grid.setCloud(*target_cloud_ptr);

  gtsam::Pose3 last_T_gtsam = gtsam::Pose3(gtsam::Rot3(result_pose.rotation()), gtsam::Point3(result_pose.translation()));
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);

  const gtsam::Key key1 = gtsam::symbol_shorthand::X(0);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;

  std::vector<Eigen::Vector3i> nearby_grids_indices = getNearbyGridsIndices(config.num_residual_per_point);
  if (nearby_grids_indices.empty()) {
    LOG(ERROR) << "Invalid num_residual_per_point: " << config.num_residual_per_point;
    return;
  }

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt(last_T_gtsam.matrix());
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<GtsamNDTFactor *> cost_functors(source_points_transformed->size() * nearby_grids_indices.size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
      Eigen::Vector3d p_trans(
        source_points_transformed->at(idx).x,
        source_points_transformed->at(idx).y,
        source_points_transformed->at(idx).z);
      Eigen::Vector3i key = target_grid.getVoxelIndex(p_trans);
      for (int i = 0; i < nearby_grids_indices.size(); ++i) {
        Eigen::Vector3i voxel_idx = key + nearby_grids_indices[i];
        auto* voxel = target_grid.getVoxel(voxel_idx);
        if (!voxel) continue;
        Eigen::Vector3d d = p_trans - voxel->mean;
        double cost = d.transpose() * voxel->information * d;
        if (std::abs(cost) > config.residual_outlier_threshold) continue;
        cost_functors[idx * nearby_grids_indices.size() + i] = new GtsamNDTFactor(key1, source_cloud_ptr->at(idx), voxel->mean, voxel->information, noise_model);
      }
    });

    gtsam::NonlinearFactorGraph graph;
    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      graph.add(*cost_functor);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key1, last_T_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);
    optimizer.optimize();

    auto result = optimizer.values();
    gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key1);

    if (
      (last_T_gtsam.rotation().toQuaternion().coeffs() -
       T_result.rotation().toQuaternion().coeffs()).norm() < config.translation_eps &&
      (last_T_gtsam.translation() - T_result.translation()).norm() < config.rotation_eps) {
      break;
    }
    last_T_gtsam = T_result;
  }
  num_iterations = iterations;
  result_pose = Eigen::Isometry3d(last_T_gtsam.matrix());
}

template <typename PointT>
void NDT_GTSAM_SO3_R3(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  // 构建目标点云的NDT体素
  VoxelGrid target_grid(config.resolution);
  target_grid.setCloud(*target_cloud_ptr);

  gtsam::Rot3 last_R_gtsam = gtsam::Rot3(result_pose.rotation());
  gtsam::Point3 last_t_gtsam = gtsam::Point3(result_pose.translation());
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);

  const gtsam::Key key1 = gtsam::symbol_shorthand::X(0);
  const gtsam::Key key2 = gtsam::symbol_shorthand::X(1);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;

  std::vector<Eigen::Vector3i> nearby_grids_indices = getNearbyGridsIndices(config.num_residual_per_point);
  if (nearby_grids_indices.empty()) {
    LOG(ERROR) << "Invalid num_residual_per_point: " << config.num_residual_per_point;
    return;
  }

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R_gtsam.matrix();
    T_opt.translation() = last_t_gtsam;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<GtsamNDTFactor2 *> cost_functors(source_points_transformed->size() * nearby_grids_indices.size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
      Eigen::Vector3d p_trans(
        source_points_transformed->at(idx).x,
        source_points_transformed->at(idx).y,
        source_points_transformed->at(idx).z);
      Eigen::Vector3i key = target_grid.getVoxelIndex(p_trans);
      for (int i = 0; i < nearby_grids_indices.size(); ++i) {
        Eigen::Vector3i voxel_idx = key + nearby_grids_indices[i];
        auto* voxel = target_grid.getVoxel(voxel_idx);
        if (!voxel) continue;
        Eigen::Vector3d d = p_trans - voxel->mean;
        double cost = d.transpose() * voxel->information * d;
        if (std::abs(cost) > config.residual_outlier_threshold) continue;
        cost_functors[idx * nearby_grids_indices.size() + i] = new GtsamNDTFactor2(key1, key2, source_cloud_ptr->at(idx), voxel->mean, voxel->information, noise_model);
      }
    });

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
      (R_result.toQuaternion().coeffs() - last_R_gtsam.toQuaternion().coeffs()).norm() < config.translation_eps &&
      (t_result - last_t_gtsam).norm() < config.rotation_eps) {
      break;
    }
    last_R_gtsam = R_result;
    last_t_gtsam = t_result;
  }
  num_iterations = iterations;
  result_pose.translation() = last_t_gtsam;
  result_pose.linear() = last_R_gtsam.matrix();
}

template <typename PointT>
void NDT_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  // 构建目标点云的NDT体素
  VoxelGrid target_grid(config.resolution);
  target_grid.setCloud(*target_cloud_ptr);

  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  Eigen::Matrix4d T = result_pose.matrix();

  std::vector<Eigen::Vector3i> nearby_grids_indices = getNearbyGridsIndices(config.num_residual_per_point);
  if (nearby_grids_indices.empty()) {
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

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
      Eigen::Vector3d p_trans(
        source_points_transformed->at(idx).x,
        source_points_transformed->at(idx).y,
        source_points_transformed->at(idx).z);
      Eigen::Vector3i key = target_grid.getVoxelIndex(p_trans);
      Eigen::Matrix<double, 6, 6> H_sum = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix<double, 6, 1> b_sum = Eigen::Matrix<double, 6, 1>::Zero();
      for (const auto& nearby_idx : nearby_grids_indices) {
        Eigen::Vector3i voxel_idx = key + nearby_idx;
        auto* voxel = target_grid.getVoxel(voxel_idx);
        if (!voxel) continue;
        Eigen::Vector3d d = p_trans - voxel->mean;
        double cost = d.transpose() * voxel->information * d;
        //! 注意区分 continue 和 return
        if (std::abs(cost) > config.residual_outlier_threshold) continue;

        num_valid_points.fetch_add(1);

        Eigen::Vector3d source_point(
          source_cloud_ptr->at(idx).x,
          source_cloud_ptr->at(idx).y,
          source_cloud_ptr->at(idx).z);

        // 计算雅可比矩阵
        Eigen::Matrix<double, 3, 6> J;
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J.block<3, 3>(0, 3) = -T.block<3, 3>(0, 0) * Hat(source_point);

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
    if (delta.norm() < config.translation_eps) {
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
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  typename pcl::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pcl::NormalDistributionsTransform<PointT, PointT>());
  ndt->setResolution(config.resolution);
  ndt->setInputTarget(target_cloud_ptr);
  ndt->setInputSource(source_cloud_ptr);
  ndt->setMaximumIterations(config.max_iterations);
  ndt->setTransformationEpsilon(config.translation_eps);
  ndt->setEuclideanFitnessEpsilon(config.translation_eps);

  pcl::PointCloud<PointT> output_cloud;
  ndt->align(output_cloud, result_pose.matrix().cast<float>());

  result_pose = ndt->getFinalTransformation().template cast<double>();
  num_iterations = ndt->getFinalNumIteration();
}

template <typename PointT>
void NDT_OMP(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  typename pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<PointT, PointT>());
  ndt_omp->setResolution(config.resolution);
  ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
  ndt_omp->setInputTarget(target_cloud_ptr);
  ndt_omp->setInputSource(source_cloud_ptr);
  ndt_omp->setMaximumIterations(config.max_iterations);
  ndt_omp->setTransformationEpsilon(config.translation_eps);
  ndt_omp->setEuclideanFitnessEpsilon(config.translation_eps);

  pcl::PointCloud<PointT> output_cloud;
  ndt_omp->align(output_cloud, result_pose.matrix().cast<float>());

  result_pose = ndt_omp->getFinalTransformation().template cast<double>();
  num_iterations = ndt_omp->getFinalNumIteration();
}

template <typename PointT>
class NDTRegistration : public RegistrationBase<PointT>
{
public:
  NDTRegistration(const RegistrationConfig& config) : RegistrationBase<PointT>(config) {}

  void align(Eigen::Isometry3d& result_pose, int& num_iterations) override
  {
    result_pose = this->initial_transformation_;
    switch (this->config_.solve_type) {
      case RegistrationConfig::Ceres: {
        NDT_Ceres<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SE3: {
        NDT_GTSAM_SE3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SO3_R3: {
        NDT_GTSAM_SO3_R3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GN: {
        NDT_GN<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::PCL: {
        NDT_PCL<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::Koide: {
        NDT_OMP<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      default: {
        LOG(ERROR) << "Unknown registration solver method: " << this->config_.solve_type;
        break;
      }
    }
  }
};
