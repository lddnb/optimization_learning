/**
 * @file eskf.cpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "optimization_learning/eskf.hpp"

ESKF::ESKF() {
  Init();
}

ESKF::~ESKF() {

}

void ESKF::Init() {
  p_ = Eigen::Vector3d::Zero();
  v_ = Eigen::Vector3d::Zero();
  R_ = Eigen::Quaterniond::Identity();
  bg_ = Eigen::Vector3d::Zero();
  ba_ = Eigen::Vector3d::Zero();
  g_ = Eigen::Vector3d::Zero();
  P_ = Eigen::Matrix<double, 18, 18>::Zero();
  Q_ = Eigen::Matrix<double, 18, 18>::Zero();
}

void ESKF::Predict(Eigen::Vector3d acc_measurement, Eigen::Vector3d gyro_measurement, double dt) {
  p_ = p_ + v_ * dt + 0.5 * g_ * dt * dt + 0.5 * (R_ * (acc_measurement - ba_)) * dt * dt;
  v_ = v_ + g_ * dt + (R_ * (acc_measurement - ba_)) * dt;
  R_ = R_ * Exp((gyro_measurement - bg_) * dt);

  Eigen::Matrix<double, 18, 18> F = Eigen::Matrix<double, 18, 18>::Identity();
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
  F.block<3, 3>(3, 6) = -R_.toRotationMatrix() * Hat(acc_measurement - ba_) * dt;
  F.block<3, 3>(3, 12) = -R_.toRotationMatrix() * dt;
  F.block<3, 3>(3, 15) = Eigen::Matrix3d::Identity() * dt;
  F.block<3, 3>(6, 6) = Exp(-(gyro_measurement - bg_) * dt);
  F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;

  P_ = F * P_ * F + Q_;
}

void ESKF::Update(Eigen::Vector3d t_measurement, Eigen::Quaterniond R_measurement, double t_noise, double R_noise) {
  Eigen::Matrix<double, 6, 18> H = Eigen::Matrix<double, 6, 18>::Zero();  // 6x18
  H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 6, 6> V = Eigen::Matrix<double, 6, 6>::Zero();  // 6x6
  Eigen::Matrix<double, 6, 1> noise_vec;
  noise_vec << t_noise, t_noise, t_noise, R_noise, R_noise, R_noise;
  V = noise_vec.asDiagonal();

  Eigen::Matrix<double, 18, 6> K = P_ * H.transpose() * (H * P_ * H.transpose() + V).inverse();  // 18x6
  Eigen::Matrix<double, 6, 1> innov = Eigen::Matrix<double, 6, 1>::Zero(); // 6x1
  innov.head<3>() = t_measurement - p_;
  innov.tail<3>() = Log((R_.inverse() * R_measurement).toRotationMatrix());

  Eigen::Matrix<double, 18, 1> delta_x = K * innov;  // 18x6 * 6x1 = 18x1
  P_ = (Eigen::Matrix<double, 18, 18>::Identity() - K * H) * P_;

  p_ += delta_x.head<3>();
  v_ += delta_x.segment<3>(3);
  R_ = R_ * Exp(delta_x.segment<3>(6));
  bg_ += delta_x.segment<3>(9);
  ba_ += delta_x.segment<3>(12);
  g_ += delta_x.segment<3>(15);
}

Eigen::Vector3d ESKF::GetPosition() const {
  return p_;
}

Eigen::Quaterniond ESKF::GetRotation() const {
  return R_;
}

Eigen::Vector3d ESKF::GetVelocity() const {
  return v_;
}

Eigen::Vector3d ESKF::GetBiasGyro() const {
  return bg_;
}

Eigen::Vector3d ESKF::GetBiasAcc() const {
  return ba_;
}

Eigen::Vector3d ESKF::GetGravity() const {
  return g_;
}

Eigen::Matrix<double, 18, 18> ESKF::GetCovariance() const {
  return P_;
}

Eigen::Matrix<double, 18, 18> ESKF::GetProcessNoise() const {
  return Q_;
}

void ESKF::SetPosition(const Eigen::Vector3d& p) {
  p_ = p;
}

void ESKF::SetVelocity(const Eigen::Vector3d& v) {
  v_ = v;
}

void ESKF::SetRotation(const Eigen::Quaterniond& R) {
  R_ = R;
}

void ESKF::SetBiasGyro(const Eigen::Vector3d& bg) {
  bg_ = bg;
}

void ESKF::SetBiasAcc(const Eigen::Vector3d& ba) {
  ba_ = ba;
}

void ESKF::SetGravity(const Eigen::Vector3d& g) {
  g_ = g;
}

void ESKF::SetCovariance(const Eigen::Matrix<double, 18, 18>& P) {
  P_ = P;
}

void ESKF::SetProcessNoise(const Eigen::Matrix<double, 18, 18>& Q) {
  Q_ = Q;
}
