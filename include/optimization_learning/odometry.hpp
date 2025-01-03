/**
 * @ Author: Your name
 * @ Create Time: 1970-01-01 08:00:00
 * @ Modified by: lddnb
 * @ Modified time: 2025-01-03 10:11:53
 * @ Description:
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"
#include "optimization_learning/gicp.hpp"
#include "optimization_learning/ndt.hpp"

class LidarOdometry : public rclcpp::Node
{
public:
  explicit LidarOdometry();
  ~LidarOdometry();

private:
  void CloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void PublishTF(const std_msgs::msg::Header& header);
  void PublishOdom(const std_msgs::msg::Header& header);
  void UpdateLocalMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& msg, const Eigen::Isometry3d& pose);
  RegistrationConfig ConfigRegistration();

  std::unique_ptr<RegistrationBase<pcl::PointXYZI>> registration;
  Eigen::Isometry3d current_pose_;
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr local_map_;
  
  // ros
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::string lidar_frame_;
  nav_msgs::msg::Path path_;
};