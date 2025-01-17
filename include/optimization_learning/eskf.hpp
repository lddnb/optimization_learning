/**
 * @file eskf.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <optimization_learning/common.hpp>

class ESKF {
public:
  ESKF();
  ~ESKF();

  void Init();
  void Predict(Eigen::Vector3d acc_measurement, Eigen::Vector3d gyro_measurement, double dt);
  void Update(Eigen::Vector3d t_measurement, Eigen::Quaterniond R_measurement, double t_noise, double R_noise);

  Eigen::Vector3d GetPosition() const;
  Eigen::Vector3d GetVelocity() const;
  Eigen::Quaterniond GetRotation() const;
  Eigen::Vector3d GetBiasGyro() const;
  Eigen::Vector3d GetBiasAcc() const;
  Eigen::Vector3d GetGravity() const;
  Eigen::Matrix<double, 18, 18> GetCovariance() const;
  Eigen::Matrix<double, 18, 18> GetProcessNoise() const;

  void SetPosition(const Eigen::Vector3d& p);
  void SetVelocity(const Eigen::Vector3d& v);
  void SetRotation(const Eigen::Quaterniond& R);
  void SetBiasGyro(const Eigen::Vector3d& bg);
  void SetBiasAcc(const Eigen::Vector3d& ba);
  void SetGravity(const Eigen::Vector3d& g);
  void SetCovariance(const Eigen::Matrix<double, 18, 18>& P);
  void SetProcessNoise(const Eigen::Matrix<double, 18, 18>& Q);

private:
  Eigen::Vector3d p_;
  Eigen::Vector3d v_;
  Eigen::Quaterniond R_; 
  Eigen::Vector3d bg_;
  Eigen::Vector3d ba_;
  Eigen::Vector3d g_;

  Eigen::Matrix<double, 18, 18> Q_;

  Eigen::Matrix<double, 18, 18> P_;

  double timestamp_;
};
