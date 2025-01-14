/**
 * @file point_to_plane_icp.hpp
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

#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>
#include "small_gicp/factors/plane_icp_factor.hpp"
#include <small_gicp/registration/registration_helper.hpp>

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

  template <typename PointT>
  CeresCostFunctorP2Plane(
    const PointT& curr_point,
    const PointT& target_point,
    const pcl::Normal& normal)
  : curr_point_(curr_point.x, curr_point.y, curr_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    normal_(normal.normal_x, normal.normal_y, normal.normal_z)
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

  template <typename PointT>
  GtsamIcpFactorP2Plane(
    gtsam::Key key,
    const PointT& source_point,
    const PointT& target_point,
    const pcl::Normal& normal,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    normal_(normal.normal_x, normal.normal_y, normal.normal_z)
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

  template <typename PointT>
  GtsamIcpFactorP2Plane2(
    gtsam::Key key1,
    gtsam::Key key2,
    const PointT& source_point,
    const PointT& target_point,
    const pcl::Normal& normal,
    const gtsam::SharedNoiseModel& cost_model)
  : gtsam::NoiseModelFactor2<gtsam::Rot3, gtsam::Point3>(cost_model, key1, key2),
    source_point_(source_point.x, source_point.y, source_point.z),
    target_point_(target_point.x, target_point.y, target_point.z),
    normal_(normal.normal_x, normal.normal_y, normal.normal_z)
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
pcl::PointCloud<pcl::Normal>::Ptr EstimateNormal(const typename pcl::PointCloud<PointT>::Ptr& cloud_ptr, const int& num_neighbors)
{
  pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
  ne.setInputCloud(cloud_ptr);
  ne.setNumberOfThreads(4);
  typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  ne.setSearchMethod(tree);
  ne.setKSearch(num_neighbors);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*normals);
  return normals;
}

template <typename PointT>
void P2PlaneICP_Ceres(
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

  // 计算target点云的法向量
  pcl::PointCloud<pcl::Normal>::Ptr target_normals =
    EstimateNormal<PointT>(target_cloud_ptr, config.num_neighbors);

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R.toRotationMatrix();
    T_opt.translation() = last_t;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<CeresCostFunctorP2Plane *> cost_functors(source_points_transformed->size(), nullptr);
    std::vector<int> index(source_points_transformed->size());
    std::iota(index.begin(), index.end(), 0);

    // 并行执行近邻搜索
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      bool valid = kdtree.nearestKSearch(source_points_transformed->at(idx), 1, nn_indices, nn_distances) > 0;

      if (!valid || nn_distances[0] > config.max_correspondence_distance) {
        return;
      }
      const auto& normal = target_normals->at(nn_indices[0]);
      if (!pcl::isFinite(normal)) {
        return;
      }

      cost_functors[idx] =
        new CeresCostFunctorP2Plane(source_cloud_ptr->at(idx), target_cloud_ptr->at(nn_indices[0]), normal);
    });

    ceres::Problem problem;
    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    problem.AddParameterBlock(T.data(), 7, se3);

    for (const auto& cost_functor : cost_functors) {
      if (cost_functor == nullptr) {
        continue;
      }
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctorP2Plane, 1, 7>(cost_functor);
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

template <typename PointT>
void P2PlaneICP_GTSAM_SE3(
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
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(1, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;

  pcl::PointCloud<pcl::Normal>::Ptr target_normals =
    EstimateNormal<PointT>(target_cloud_ptr, config.num_neighbors);

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt(last_T_gtsam.matrix());
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<std::unique_ptr<GtsamIcpFactorP2Plane>> cost_functors(source_points_transformed->size());
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
        
        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

        const auto& normal = target_normals->at(nn_indices[0]);
        if (!pcl::isFinite(normal)) {
          return;
        }

        // 构建因子
        cost_functors[idx] = std::make_unique<GtsamIcpFactorP2Plane>(
          key, 
          source_cloud_ptr->at(idx), 
          target_cloud_ptr->at(nn_indices[0]), 
          normal, 
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

template <typename PointT>
void P2PlaneICP_GTSAM_SO3_R3(
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
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(1, 1);
  gtsam::GaussNewtonParams params_gn;
  if (config.verbose) {
    params_gn.setVerbosity("ERROR");
  }
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = config.translation_eps;

  pcl::PointCloud<pcl::Normal>::Ptr target_normals =
    EstimateNormal<PointT>(target_cloud_ptr, config.num_neighbors);

  int iterations = 0;
  for (; iterations < config.max_iterations; ++iterations) {
    Eigen::Isometry3d T_opt = Eigen::Isometry3d::Identity();
    T_opt.linear() = last_R_gtsam.matrix();
    T_opt.translation() = last_t_gtsam;
    pcl::transformPointCloud(*source_cloud_ptr, *source_points_transformed, T_opt.matrix());

    std::vector<std::unique_ptr<GtsamIcpFactorP2Plane2>> cost_functors(source_points_transformed->size());
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
        
        if (!valid || nn_distances[0] > config.max_correspondence_distance) {
          return;
        }

        const auto& normal = target_normals->at(nn_indices[0]);
        if (!pcl::isFinite(normal)) {
          return;
        }

        // 构建因子
        cost_functors[idx] = std::make_unique<GtsamIcpFactorP2Plane2>(
          key, 
          key2,
          source_cloud_ptr->at(idx), 
          target_cloud_ptr->at(nn_indices[0]), 
          normal, 
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
  }
  num_iterations = iterations;
  result_pose.translation() = last_t_gtsam;
  result_pose.linear() = last_R_gtsam.matrix();
}

// Gauss-Newton's method solve NICP.
template <typename PointT>
void P2PlaneICP_GN(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  typename pcl::PointCloud<PointT>::Ptr source_points_transformed(new pcl::PointCloud<PointT>);
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(target_cloud_ptr);

  pcl::PointCloud<pcl::Normal>::Ptr target_normals =
    EstimateNormal<PointT>(target_cloud_ptr, config.num_neighbors);

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

        const auto& normal = target_normals->at(nn_indices[0]);
        if (!pcl::isFinite(normal)) {
          return;
        }

        const Eigen::Vector3d normal_vector = normal.getNormalVector3fMap().cast<double>();

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
        double error = normal_vector.transpose() * (curr_point - target_point);

        Eigen::Matrix<double, 1, 6> Jacobian = Eigen::Matrix<double, 1, 6>::Zero();
        // 构建雅克比矩阵
        Jacobian.leftCols(3) = normal_vector.transpose();
        Jacobian.rightCols(3) = -normal_vector.transpose() * last_T.block<3, 3>(0, 0) * Hat(source_point);
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


template <typename PointT>
void P2PlaneICP_PCL(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Isometry3d& result_pose,
  int& num_iterations,
  const RegistrationConfig& config)
{
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr source_cloud_with_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_cloud_with_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
  pcl::NormalEstimationOMP<PointT, pcl::Normal> norm_est;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
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
  nicp.setTransformationEpsilon(config.translation_eps);
  nicp.setEuclideanFitnessEpsilon(config.translation_eps);
  nicp.setMaximumIterations(config.max_iterations);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZINormal>);
  nicp.align(*aligned, result_pose.matrix().cast<float>());
  
  result_pose = nicp.getFinalTransformation().cast<double>();
  num_iterations = nicp.nr_iterations_;
}


// small_gicp point-to-plane ICP
template <typename PointT>
void P2PlaneICP_small_gicp(
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
    [](const PointT& point) { return Eigen::Vector3d(point.x, point.y, point.z); });
  std::transform(
    std::execution::par,
    target_cloud_ptr->begin(),
    target_cloud_ptr->end(),
    target_eigen.begin(),
    [](const PointT& point) { return Eigen::Vector3d(point.x, point.y, point.z); });

  auto target = std::make_shared<small_gicp::PointCloud>(target_eigen);
  auto source = std::make_shared<small_gicp::PointCloud>(source_eigen);

  // Create KdTree
  auto target_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
    target,
    small_gicp::KdTreeBuilderOMP(config.num_threads));

  // Estimate point covariances
  estimate_normals_omp(*target, *target_tree, config.num_neighbors, config.num_threads);

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

template <typename PointT>
class NICPRegistration : public RegistrationBase<PointT>
{
public:
  NICPRegistration(const RegistrationConfig& config) : RegistrationBase<PointT>(config) {}

  void align(Eigen::Isometry3d& result_pose, int& num_iterations) override
  {
    result_pose = this->initial_transformation_;
    switch (this->config_.solve_type) {
      case RegistrationConfig::Ceres: {
        P2PlaneICP_Ceres<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SE3: {
        P2PlaneICP_GTSAM_SE3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GTSAM_SO3_R3: {
        P2PlaneICP_GTSAM_SO3_R3<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::GN: {
        P2PlaneICP_GN<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::PCL: {
        P2PlaneICP_PCL<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      case RegistrationConfig::Koide: {
        P2PlaneICP_small_gicp<PointT>(this->source_cloud_, this->target_cloud_, result_pose, num_iterations, this->config_);
        break;
      }
      default: {
        LOG(ERROR) << "Unknown registration solver method: " << this->config_.solve_type;
        break;
      }
    }
  }
};