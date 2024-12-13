/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-13 11:52:30
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-13 16:38:11
 * @ Description:
 */

#pragma once

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include "so3_tool.hpp"

// 旋转残差
class CostFunctor1
{
public:
  CostFunctor1(const Eigen::Matrix3d& R_) : R0(R_.transpose()) {}
  CostFunctor1(const Eigen::Quaterniond& R_) : R0(R_.inverse()) {}

  template <typename T>
  bool operator()(const T* const R_, T* residual_) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> R(R_);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residual_);

    Eigen::Quaternion<T> q_err = R0.cast<T>() * R;

    // 使用ceres::QuaternionToAngleAxis
    Eigen::Matrix<T, 4, 1> ceres_q_err{q_err.w(), q_err.x(), q_err.y(), q_err.z()};
    ceres::QuaternionToAngleAxis(ceres_q_err.data(), residual.data());

    return true;
  }

private:
  Eigen::Quaterniond R0;
};

class CostFunctor2
{
public:
  CostFunctor2(const Eigen::Matrix3d& R) : R0(R.transpose()) {}
  CostFunctor2(const Eigen::Quaterniond& R) : R0(R.inverse()) {}

  template <typename T>
  bool operator()(const T* const q, T* residual) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> q_eigen(q);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_eigen(residual);

    Eigen::Quaternion<T> R_delta = R0.cast<T>() * q_eigen;

    // 使用 Eigen::AngleAxis
    auto residual_vec = Eigen::AngleAxis<T>(R_delta);
    residual_eigen = residual_vec.angle() * residual_vec.axis();
    return true;
  }

private:
  Eigen::Quaterniond R0;
};

// ceres 手动求导
// Todo: 这个求导方式有问题，待修复，详见 Issue #1
class MyCostFunction : public ceres::SizedCostFunction<3, 4>
{
public:
  MyCostFunction(const Eigen::Quaterniond& R_) : R0(R_.inverse()) {}

  virtual bool Evaluate(double const* const* q, double* residuals, double** jacobians) const override
  {
    Eigen::Map<const Eigen::Quaterniond> q_eigen(q[0]);
    Eigen::Map<Eigen::Vector3d> residual_eigen(residuals);

    Eigen::Quaterniond R_delta = R0 * q_eigen;

    // 残差定义 1
    auto residual_vec = Eigen::AngleAxisd(R_delta);
    residual_eigen = residual_vec.angle() * residual_vec.axis();

    // 残差定义 2
    // 参考的 https://fzheng.me/2018/05/22/quaternion-matrix-so3-jacobians/
    // residual_eigen = 2.0 * R_delta.vec();

    if (jacobians == nullptr) {
      return true;
    }
    if (jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian_R(jacobians[0]);
      jacobian_R.setZero();

      // 残差 1 的雅克比
      Eigen::Vector3d residual = residual_eigen;
      jacobian_R.leftCols(3) = Jacob_right_inv(residual);

      // 残差 2 的雅克比
      // jacobian_R  << R_delta.w(), R_delta.z(), -R_delta.y(), -R_delta.x(),
      //    -R_delta.z(), R_delta.w(), R_delta.x(), -R_delta.y(),
      //    R_delta.y(), -R_delta.x(), R_delta.w(), -R_delta.z();
      // jacobian_R.leftCols(3) << R_delta.w() * Eigen::Matrix3d::Identity() + Hat(R_delta.vec());
    }
    return true;
  }

private:
  Eigen::Quaterniond R0;
};

class GtsamFactor : public gtsam::NoiseModelFactor1<gtsam::Rot3>
{
public:
  GtsamFactor(gtsam::Key key, const Eigen::Quaterniond& R, const gtsam::SharedNoiseModel& model)
  : gtsam::NoiseModelFactor1<gtsam::Rot3>(model, key),
    R0(R.inverse())
  {
  }

  virtual gtsam::Vector evaluateError(const gtsam::Rot3& R, boost::optional<gtsam::Matrix&> H = boost::none)
    const override
  {
    gtsam::Rot3 R_delta = R0 * R;
    gtsam::Vector residual = gtsam::Rot3::Logmap(R_delta);
    if (H) {
      *H = gtsam::I_3x3;
    }
    return residual;
  }

private:
  gtsam::Rot3 R0;
};