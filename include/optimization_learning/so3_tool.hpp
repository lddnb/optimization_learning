/**
 * @file so3_tool.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> Hat(const Eigen::MatrixBase<Derived>& mat)
{
  Eigen::Matrix<typename Derived::Scalar, 3, 3> mat_skew;
  mat_skew << typename Derived::Scalar(0), -mat(2), mat(1), mat(2), typename Derived::Scalar(0), -mat(0), -mat(1),
    mat(0), typename Derived::Scalar(0);
  return mat_skew;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> Exp(const Eigen::MatrixBase<Derived>& ang)
{
  typename Derived::Scalar ang_norm = ang.norm();
  Eigen::Matrix<typename Derived::Scalar, 3, 3> Eye3 = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
  if (ang_norm > 0.0000001) {
    Eigen::Matrix<typename Derived::Scalar, 3, 1> r_axis = ang / ang_norm;
    Eigen::Matrix<typename Derived::Scalar, 3, 3> K = Hat(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  } else {
    return Eye3;
  }
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> Jacob_right_inv(const Eigen::MatrixBase<Derived>& vec)
{
  Eigen::Matrix<typename Derived::Scalar, 3, 3> hat_v = Hat(vec);
  Eigen::Matrix<typename Derived::Scalar, 3, 3> res = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
  if (vec.norm() > 0.0000001) {
    res =
      Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + 0.5 * hat_v +
      (1 - vec.norm() * std::cos(vec.norm() / 2) / 2 / std::sin(vec.norm() / 2)) * hat_v * hat_v / vec.squaredNorm();
  } else {
    res = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
  }
  return res;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived>& q)
{
  Eigen::Matrix<typename Derived::Scalar, 4, 4> ans = Eigen::Matrix<typename Derived::Scalar, 4, 4>::Zero();
  ans << q.w(), -q.x(), -q.y(), -q.z(),
         q.x(), q.w(), -q.z(), q.y(),
         q.y(), q.z(), q.w(), -q.x(),
         q.z(), -q.y(), q.x(), q.w();
  return ans;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> QRight(const Eigen::QuaternionBase<Derived>& q)
{
  Eigen::Matrix<typename Derived::Scalar, 4, 4> ans = Eigen::Matrix<typename Derived::Scalar, 4, 4>::Zero();
  ans << q.w(), -q.x(), -q.y(), -q.z(),
         q.x(), q.w(), q.z(), -q.y(),
         q.y(), -q.z(), q.w(), q.x(),
         q.z(), q.y(), -q.x(), q.w();
  return ans;
}

// 右扰动模型的四元数
class RightQuaternionManifold : public ceres::Manifold
{
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
  {
    // 流型加法，x boxplus delta = x * Exp(delta)
    Eigen::Map<const Eigen::Quaterniond> q(x);
    Eigen::Map<const Eigen::Vector3d> d(delta);

    Eigen::Map<Eigen::Quaterniond> q_plus_d(x_plus_delta);

    const double norm_delta = d.norm();

    if (norm_delta == 0.0) {
      q_plus_d = q;
      return true;
    }

    const double sin_delta_by_delta = (std::sin(norm_delta) / norm_delta);
    Eigen::Quaterniond dq(
      std::cos(norm_delta),
      sin_delta_by_delta * delta[0],
      sin_delta_by_delta * delta[1],
      sin_delta_by_delta * delta[2]);

    q_plus_d = (q * dq).normalized();

    return true;
  }
  bool PlusJacobian(const double* x, double* jacobian) const override
  {
    // derivative of  x * Exp(delta)  wrt. delta at delta=0
    Eigen::Map<const Eigen::Quaterniond> q(x);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
    // dq * q 的左扰动 Jacobian
    // j << q.w(), q.z(), -q.y(),
    //      -q.z(), q.w(), q.x(),
    //      q.y(), -q.x(), q.w(),
    //      -q.x(), -q.y(), -q.z();
    
    // q * dq 的右扰动 Jacobian
    // 这里和Qleft有区别主要是 w 的位置不同
    // Eigen中四元数 x,y,z,w, 实部在后面，所以位置需要调整
    j << q.w(), -q.z(), q.y(),
         q.z(), q.w(), -q.x(),
         -q.y(), q.x(), q.w(),
         -q.x(), -q.y(), -q.z();

    // j.topRows<3>().setIdentity();
    // j.bottomRows<1>().setZero();

    return true;
  }
  bool Minus(const double* y, const double* x, double* y_minus_x) const override
  {
    // 流型减法，y boxminus x = Log(x^{-1} * y)
    Eigen::Map<const Eigen::Quaterniond> q_y(y);
    Eigen::Map<const Eigen::Quaterniond> q_x(x);

    Eigen::Map<Eigen::Vector3d> vec_delta(y_minus_x);

    Eigen::Quaterniond q_delta = (q_x.inverse() * q_y).normalized();
    auto axis_angle = Eigen::AngleAxisd(q_delta);
    vec_delta = axis_angle.axis() * axis_angle.angle();

    // 也可以使用 ceres::QuaternionToAngleAxis
    // Eigen::Vector4d ceres_q_err{q_delta.w(), q_delta.x(), q_delta.y(), q_delta.z()};
    // ceres::QuaternionToAngleAxis(ceres_q_err.data(), vec_delta.data());

    return true;
  }
  bool MinusJacobian(const double* x, double* jacobian) const override
  {
    // derivative of Log(x^{-1} * y) by y at y=x
    Eigen::Map<const Eigen::Quaterniond> q(x);
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> j(jacobian);
    // 左扰动的 Jacobian
    // j << q.w(), -q.z(), -q.y(), -q.x(),
    //      q.z(), q.w(), -q.x(), -q.y(),
    //      -q.y(), q.x(), q.w(), -q.z();
    
    // 右扰动的 Jacobian
    j << q.w(),  q.z(), -q.y(), -q.x(),
        -q.z(),  q.w(),  q.x(), -q.y(),
         q.y(), -q.x(),  q.w(), -q.z();

    // j.leftCols(3) = Eigen::Matrix3d::Identity();
    // j.rightCols(1).setZero();

    return true;
  }
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 3; }
};

