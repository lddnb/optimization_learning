/**
 * @file imu_integration.cpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "optimization_learning/imu_integration.hpp"

ImuIntegration::ImuIntegration()
{
  current_pose_ = Eigen::Isometry3d::Identity();
  current_velocity_ = Eigen::Vector3d::Zero();
  current_timestamp_ = 0.0;

  gravity_ = Eigen::Vector3d(0.0, 0.0, -9.81);
  bias_acc_ = Eigen::Vector3d::Zero();
  bias_gyr_ = Eigen::Vector3d::Zero();

  is_initialized_ = false;
  init_imu_size_ = 0;
}

ImuIntegration::~ImuIntegration()
{
}

void ImuIntegration::ProcessImu(
  const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer,
  const pcl::PointCloud<PointType>::Ptr& input_cloud,
  pcl::PointCloud<PointType>::Ptr& output_cloud)
{
  if (!is_initialized_) {
    Init(imu_buffer);
  }
}

void ImuIntegration::Init(const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer)
{
  static Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();
  static Eigen::Vector3d mean_gyr = Eigen::Vector3d::Zero();
  static Eigen::Vector3d cur_acc = Eigen::Vector3d::Zero();
  static Eigen::Vector3d cur_gyr = Eigen::Vector3d::Zero();
  if (mean_acc.isZero() && mean_gyr.isZero()) {
    cur_acc << imu_buffer.front()->linear_acceleration.x, imu_buffer.front()->linear_acceleration.y,
      imu_buffer.front()->linear_acceleration.z;
    cur_gyr << imu_buffer.front()->angular_velocity.x, imu_buffer.front()->angular_velocity.y,
      imu_buffer.front()->angular_velocity.z;
    mean_acc = cur_acc;
    mean_gyr = cur_gyr;
    init_imu_size_++;
  }
  for (const auto& imu : imu_buffer) {
    cur_acc << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
    cur_gyr << imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z;
    mean_acc += (cur_acc - mean_acc) / init_imu_size_;
    mean_gyr += (cur_gyr - mean_gyr) / init_imu_size_;

    init_imu_size_++;
  }
  if (init_imu_size_ >= 100) {
    bias_acc_ = mean_acc;
    bias_gyr_ = mean_gyr;
    gravity_ = -mean_acc / mean_acc.norm() * 9.81;
    last_imu_msg_ = imu_buffer.back();
    is_initialized_ = true;
  }
}

// todo
void ImuIntegration::UndistortPoint(
  const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer,
  const pcl::PointCloud<PointType>::Ptr& point_cloud,
  pcl::PointCloud<PointType>::Ptr& undistorted_point_cloud)
{
  auto imu_msgs = imu_buffer;
  imu_msgs.push_front(last_imu_msg_);
  const double imu_begin_timestamp = imu_msgs.front()->header.stamp.sec + imu_msgs.front()->header.stamp.nanosec * 1e-9;
  const double imu_end_timestamp = imu_msgs.back()->header.stamp.sec + imu_msgs.back()->header.stamp.nanosec * 1e-9;
  const double start_orientation = std::atan2(point_cloud->points[0].y, point_cloud->points[0].x);

  for (size_t i = 1; i < point_cloud->size(); i++) {
    const double current_orientation = std::atan2(point_cloud->points[i].y, point_cloud->points[i].x);
    const double delta_orientation = current_orientation - start_orientation;
    const double delta_time = imu_msgs[i]->header.stamp.sec + imu_msgs[i]->header.stamp.nanosec * 1e-9 - imu_begin_timestamp;
    const Eigen::Vector3d acc = Eigen::Vector3d(imu_msgs[i]->linear_acceleration.x, imu_msgs[i]->linear_acceleration.y, imu_msgs[i]->linear_acceleration.z);
    const Eigen::Vector3d gyr = Eigen::Vector3d(imu_msgs[i]->angular_velocity.x, imu_msgs[i]->angular_velocity.y, imu_msgs[i]->angular_velocity.z);
  }
}

void ImuIntegration::Integrate(const sensor_msgs::msg::Imu::SharedPtr& msg)
{
  const double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  const Eigen::Vector3d acc = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  const Eigen::Vector3d gyr = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  if (current_timestamp_ == 0.0) {
    current_timestamp_ = timestamp;
    return;
  }

  const double dt = timestamp - current_timestamp_;
  if (dt <= 0.0 || dt > 0.1) {
    return;
  }

  const Eigen::Vector3d acc_body = acc - bias_acc_;
  const Eigen::Vector3d gyr_body = gyr - bias_gyr_;

  Eigen::Vector3d p_ = current_pose_.translation();
  Eigen::Vector3d v_ = current_velocity_;
  Eigen::Quaterniond R_ = Eigen::Quaterniond(current_pose_.rotation());

  p_ = p_ + v_ * dt + 0.5 * gravity_ * dt * dt + 0.5 * (R_ * acc_body) * dt * dt;
  v_ = v_ + gravity_ * dt + R_ * acc_body * dt;
  R_ = R_ * Exp(gyr_body * dt);

  current_pose_.translation() = p_;
  current_pose_.linear() = R_.toRotationMatrix();
  current_velocity_ = v_;
  current_timestamp_ = timestamp;
}

void ImuIntegration::UpdatePose(const Eigen::Isometry3d& pose)
{
  current_pose_ = pose;
}

Eigen::Isometry3d ImuIntegration::GetCurrentPose() const
{
  return current_pose_;
}

Eigen::Vector3d ImuIntegration::GetCurrentVelocity() const
{
  return current_velocity_;
}

void ImuIntegration::SetBias(const Eigen::Vector3d& bias_acc, const Eigen::Vector3d& bias_gyr)
{
  bias_acc_ = bias_acc;
  bias_gyr_ = bias_gyr;
}