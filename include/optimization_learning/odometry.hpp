/**
 * @file odometry.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"
#include "optimization_learning/gicp.hpp"
#include "optimization_learning/ndt.hpp"
#include "optimization_learning/imu_integration.hpp"
#include "optimization_learning/cloud_process.hpp"

class LidarOdometry : public rclcpp::Node
{
public:
  explicit LidarOdometry();
  ~LidarOdometry();

private:
  void CloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void GroundTruthPathCallback(const nav_msgs::msg::Path::SharedPtr msg);
  bool GetSyncedData(SyncedData& synced_data);
  void PublishTF(const builtin_interfaces::msg::Time& timestamp);
  void PublishOdom(const builtin_interfaces::msg::Time& timestamp);
  void UpdateLocalMap(const pcl::PointCloud<PointType>::Ptr& msg, const Eigen::Isometry3d& pose);
  void SaveMappingResult();
  RegistrationConfig ConfigRegistration();

  void MainThread();

  std::unique_ptr<ImuIntegration> imu_integration_;
  std::unique_ptr<RegistrationBase<PointType>> registration;
  Eigen::Isometry3d current_pose_;
  pcl::VoxelGrid<PointType> voxel_grid_;
  pcl::PointCloud<PointType>::Ptr local_map_;
  
  // ros
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr ground_truth_path_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::string lidar_frame_;
  nav_msgs::msg::Path path_;
  nav_msgs::msg::Path ground_truth_path_;

  // local map config
  int local_map_min_frame_size_;
  int update_frame_size_;
  double update_translation_delta_;
  double update_rotation_delta_;

  // save map
  std::string save_map_path_;

  // time eval
  TimeEval downsample_time_eval_;
  TimeEval registration_time_eval_;

  // thread
  std::unique_ptr<std::thread> main_thread_;
  std::mutex cloud_buffer_mutex_;
  std::mutex imu_buffer_mutex_;

  std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> lidar_cloud_buffer_;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
  SyncedData synced_data_;

  // calibration
  Eigen::Isometry3d T_imu2lidar_;

  // eskf
  std::unique_ptr<ESKF> eskf_;
};
