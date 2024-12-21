/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-19 15:04:38
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-20 18:57:10
 * @ Description:
 */

#pragma once
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include <optimization_learning/so3_tool.hpp>

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

// Gauss-Newton's method solve NICP.
template <typename PointT>
bool MatchP2Plane(
  const typename pcl::PointCloud<PointT>::Ptr& source_cloud_ptr,
  const Eigen::Matrix4d& predict_pose,
  const typename pcl::PointCloud<PointT>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose)
{
  bool has_converge_ = false;
  int max_iterations_ = 30;
  double max_correspond_distance_ = 0.5;
  double transformation_epsilon_ = 1e-3;

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
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree_flann_ptr_->nearestKSearch(transformed_point, 5, nn_indices, nn_distances);

      std::vector<Eigen::Vector3d> plane_points;
      for (size_t i = 0; i < 5; ++i) {
        plane_points.emplace_back(
          target_cloud_ptr->at(nn_indices[i]).x,
          target_cloud_ptr->at(nn_indices[i]).y,
          target_cloud_ptr->at(nn_indices[i]).z);
      }
      Eigen::Matrix<double, 4, 1> plane_coeffs;
       if (nn_distances[0] > 1 || !FitPlane(plane_points, plane_coeffs)) {
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