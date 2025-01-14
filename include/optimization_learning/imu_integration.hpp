/**
 * @file imu_integration.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <Eigen/Eigen>
#include <sensor_msgs/msg/imu.hpp>

#include "optimization_learning/common.hpp"

using PointType = pcl::PointXYZ;

class ImuIntegration
{
public:
  ImuIntegration();
  ~ImuIntegration();

  void ProcessImu(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer,
    const pcl::PointCloud<PointType>::Ptr& input_cloud,
    pcl::PointCloud<PointType>::Ptr& output_cloud);
  void Init(const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer);

  void UndistortPoint(
    const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer,
    const pcl::PointCloud<PointType>::Ptr& point_cloud,
    pcl::PointCloud<PointType>::Ptr& undistorted_point_cloud);
  void Integrate(const sensor_msgs::msg::Imu::SharedPtr& msg);
  void UpdatePose(const Eigen::Isometry3d& pose);
  Eigen::Isometry3d GetCurrentPose() const;
  Eigen::Vector3d GetCurrentVelocity() const;
  void SetBias(const Eigen::Vector3d& bias_acc, const Eigen::Vector3d& bias_gyr);

private:
  Eigen::Isometry3d current_pose_;
  Eigen::Vector3d current_velocity_;
  double current_timestamp_;

  Eigen::Vector3d gravity_;
  Eigen::Vector3d bias_acc_;
  Eigen::Vector3d bias_gyr_;

  sensor_msgs::msg::Imu::SharedPtr last_imu_msg_;

  // init
  bool is_initialized_;
  int init_imu_size_;
};
