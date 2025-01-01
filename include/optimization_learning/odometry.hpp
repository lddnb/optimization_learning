/**
 * @ Author: Your name
 * @ Create Time: 1970-01-01 08:00:00
 * @ Modified by: Your name
 * @ Modified time: 2025-01-01 17:36:27
 * @ Description:
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"
#include "optimization_learning/gicp.hpp"
#include "optimization_learning/ndt.hpp"

class LidarOdometry : public rclcpp::Node
{
public:
  explicit LidarOdometry(const RegistrationConfig& config);
  ~LidarOdometry();

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr previous_cloud_;
  std::unique_ptr<RegistrationBase<pcl::PointXYZI>> registration;
  Eigen::Isometry3d current_pose_;
};