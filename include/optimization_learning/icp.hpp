/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-13 14:47:47
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-19 18:25:34
 * @ Description:
 */

#pragma once

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <pcl/point_types.h>

#include <optimization_learning/so3_tool.hpp>

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
