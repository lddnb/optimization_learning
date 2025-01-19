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
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "optimization_learning/common.hpp"
#include "optimization_learning/eskf.hpp"

using PointType = PointXYZIT;

template <typename PointT>
const bool time_list(const PointT& x, const PointT& y) {return (x.time < y.time);};

struct SyncedData {
  pcl::PointCloud<PointType>::Ptr cloud;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu;
  double lidar_begin_timestamp;
  double lidar_end_timestamp;
};

class ImuIntegration
{
public:
  ImuIntegration();
  ~ImuIntegration();

  bool ProcessImu(
    const SyncedData& synced_data,
    pcl::PointCloud<PointType>& output_cloud,
    std::unique_ptr<ESKF>& eskf);
  void Init(const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer, std::unique_ptr<ESKF>& eskf);

  void UndistortPoint(
    const SyncedData& synced_data,
    pcl::PointCloud<PointType>& undistorted_point_cloud,
    std::unique_ptr<ESKF>& eskf);
  void Integrate(const sensor_msgs::msg::Imu::SharedPtr& msg);
  void UpdatePose(const Eigen::Isometry3d& pose);
  Eigen::Isometry3d GetCurrentPose() const;
  Eigen::Vector3d GetCurrentVelocity() const;
  void SetBias(const Eigen::Vector3d& bias_acc, const Eigen::Vector3d& bias_gyr);
  int GetInitImuSize() const;

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
  std::deque<sensor_msgs::msg::Imu::SharedPtr> init_imu_deque_;

  Eigen::Vector3d mean_acc_;
  Eigen::Vector3d mean_gyr_;
};
