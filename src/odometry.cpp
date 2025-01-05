/**
 * @file odometry.cpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "optimization_learning/odometry.hpp"

LidarOdometry::LidarOdometry() : Node("lidar_odometry")
{
  LOG(INFO) << "LidarOdometry Initializing";

  ConfigRegistration();

  // 设置QoS
  rclcpp::QoS qos(10);
  qos.best_effort();
  qos.durability_volatile();
  qos.keep_last(10);

  // 创建订阅者和发布者

  this->declare_parameter("lidar_topic", "/sensing/lidar/top/rectified/pointcloud");
  this->declare_parameter("output_odom_topic", "/output_odom");
  this->declare_parameter("output_path_topic", "/output_path");
  this->declare_parameter("output_cloud_topic", "/output_cloud");
  const auto lidar_topic = this->get_parameter("lidar_topic").as_string();
  const auto odom_topic = this->get_parameter("output_odom_topic").as_string();
  const auto path_topic = this->get_parameter("output_path_topic").as_string();
  const auto cloud_topic = this->get_parameter("output_cloud_topic").as_string();
  LOG(INFO) << "[cfg] lidar_topic: " << lidar_topic;

  this->declare_parameter("local_map_min_frame_size", 3);
  this->declare_parameter("update_frame_size", 10);
  this->declare_parameter("update_translation_delta", 2.0);
  this->declare_parameter("update_rotation_delta", 30.0);
  local_map_min_frame_size_ = this->get_parameter("local_map_min_frame_size").as_int();
  update_frame_size_ = this->get_parameter("update_frame_size").as_int();
  update_translation_delta_ = this->get_parameter("update_translation_delta").as_double();
  update_rotation_delta_ = this->get_parameter("update_rotation_delta").as_double();
  LOG(INFO) << "[cfg] local_map_min_frame_size: " << local_map_min_frame_size_;
  LOG(INFO) << "[cfg] update_frame_size: " << update_frame_size_;
  LOG(INFO) << "[cfg] update_translation_delta: " << update_translation_delta_;
  LOG(INFO) << "[cfg] update_rotation_delta: " << update_rotation_delta_;
  update_rotation_delta_ = update_rotation_delta_ * M_PI / 180;

  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    lidar_topic, qos,
    std::bind(&LidarOdometry::CloudCallback, this, std::placeholders::_1));
  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
  path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic, 10);
  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic, 10);
  
  current_pose_ = Eigen::Isometry3d::Identity();
  LOG(INFO) << "LidarOdometry Initialized";

  // 初始化tf广播器
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
}

LidarOdometry::~LidarOdometry() {}

RegistrationConfig LidarOdometry::ConfigRegistration()
{
  // 声明并获取参数
  this->declare_parameter("registration_type", 3);  // Default: NDT
  this->declare_parameter("solve_type", 5);        // Default: OMP
  this->declare_parameter("max_correspondence_distance", 1.0);
  this->declare_parameter("max_iterations", 30);
  this->declare_parameter("translation_epsilon", 0.01);
  this->declare_parameter("rotation_epsilon", 0.01);
  this->declare_parameter("resolution", 1.0);
  this->declare_parameter("num_threads", 4);
  this->declare_parameter("downsample_leaf_size", 0.1);
  this->declare_parameter("verbose", false);

  // 读取参数
  RegistrationConfig reg_config;
  reg_config.registration_type = static_cast<RegistrationConfig::RegistrationType>(
    this->get_parameter("registration_type").as_int());
  reg_config.solve_type = static_cast<RegistrationConfig::SolveType>(
    this->get_parameter("solve_type").as_int());
  reg_config.max_correspondence_distance = this->get_parameter("max_correspondence_distance").as_double();
  reg_config.max_iterations = this->get_parameter("max_iterations").as_int();
  reg_config.translation_eps = this->get_parameter("translation_epsilon").as_double();
  reg_config.rotation_eps = this->get_parameter("rotation_epsilon").as_double();
  reg_config.resolution = this->get_parameter("resolution").as_double();
  reg_config.num_threads = this->get_parameter("num_threads").as_int();
  reg_config.verbose = this->get_parameter("verbose").as_bool();
  reg_config.downsample_leaf_size = this->get_parameter("downsample_leaf_size").as_double();

  // 创建配准对象
  switch (reg_config.registration_type) {
    case RegistrationConfig::ICP: {
      registration = std::make_unique<ICPRegistration<pcl::PointXYZI>>(reg_config);
      break;
    }
    case RegistrationConfig::NICP: {
      registration = std::make_unique<NICPRegistration<pcl::PointXYZI>>(reg_config);
      break;
    }
    case RegistrationConfig::GICP: {
      registration = std::make_unique<GICPRegistration<pcl::PointXYZI>>(reg_config);
      break;
    }
    case RegistrationConfig::NDT: {
      registration = std::make_unique<NDTRegistration<pcl::PointXYZI>>(reg_config);
      break;
    }
    default: {
      LOG(FATAL) << "Unknown registration type";
      exit(1);
    }
  }

  voxel_grid_.setLeafSize(reg_config.downsample_leaf_size, reg_config.downsample_leaf_size, reg_config.downsample_leaf_size);

  LOG(INFO) << "[cfg] registration_type: " << reg_config.registration_type;
  LOG(INFO) << "[cfg] solve_type: " << reg_config.solve_type;

  return reg_config;
}

void LidarOdometry::CloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  if (lidar_frame_.empty()) {
    lidar_frame_ = msg->header.frame_id;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *cloud);

  // 下采样
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  voxel_grid_.setInputCloud(cloud);
  voxel_grid_.filter(*downsampled_cloud);

  if (local_map_ == nullptr) {
    local_map_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    local_map_ = downsampled_cloud;
    return;
  }

  registration->setInputSource(downsampled_cloud);
  registration->setInputTarget(local_map_);
  registration->setInitialTransformation(current_pose_);
  int iterations = 0;
  registration->align(current_pose_, iterations);

  // 发布pose和tf
  PublishTF(msg->header);
  PublishOdom(msg->header);

  pcl::transformPointCloud(*downsampled_cloud, *downsampled_cloud, current_pose_.matrix());
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*downsampled_cloud, cloud_msg);
  //! 记得设置 frame_id
  cloud_msg.header.frame_id = "map";
  cloud_msg.header.stamp = this->now();
  cloud_pub_->publish(cloud_msg);

  UpdateLocalMap(downsampled_cloud, current_pose_);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  LOG(INFO) << "Elapsed time: " << duration << " ms, iterations: " << iterations;
}

void LidarOdometry::UpdateLocalMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& msg, const Eigen::Isometry3d& pose)
{
  static std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> local_map_list;
  static Eigen::Isometry3d last_key_pose = Eigen::Isometry3d::Identity();
  local_map_list.push_back(msg);

  auto delta_pose = pose * last_key_pose.inverse();
  if (
    local_map_list.size() >= local_map_min_frame_size_ &&
    (local_map_list.size() >= update_frame_size_ ||
     delta_pose.translation().norm() > update_translation_delta_ ||
     Eigen::AngleAxisd(delta_pose.rotation()).angle() > update_rotation_delta_)) {
    local_map_.reset(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto& cloud : local_map_list) {
      *local_map_ += *cloud;
    }
    LOG(INFO) << "Update local map size: " << local_map_list.size();
    local_map_list.clear();
    last_key_pose = pose;
    return;
  }

  *local_map_ += *msg;
}

void LidarOdometry::PublishTF(const std_msgs::msg::Header& header)
{
  geometry_msgs::msg::TransformStamped transform_stamped;
  
  // 设置时间戳稍早于点云消息
  transform_stamped.header.stamp = this->now();  // 使用当前节点的时间
  transform_stamped.header.frame_id = "map";
  transform_stamped.child_frame_id = lidar_frame_;
  
  // 设置平移
  transform_stamped.transform.translation.x = current_pose_.translation().x();
  transform_stamped.transform.translation.y = current_pose_.translation().y();
  transform_stamped.transform.translation.z = current_pose_.translation().z();
  
  // 设置旋转
  Eigen::Quaterniond quat(current_pose_.rotation());
  transform_stamped.transform.rotation.x = quat.x();
  transform_stamped.transform.rotation.y = quat.y();
  transform_stamped.transform.rotation.z = quat.z();
  transform_stamped.transform.rotation.w = quat.w();
  
  // 发布tf
  tf_broadcaster_->sendTransform(transform_stamped);
}

void LidarOdometry::PublishOdom(const std_msgs::msg::Header& header)
{
  // 发布pose
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = header;
  pose_msg.header.frame_id = "map";
  pose_msg.pose.position.x = current_pose_.translation().x();
  pose_msg.pose.position.y = current_pose_.translation().y();
  pose_msg.pose.position.z = current_pose_.translation().z();
  
  Eigen::Quaterniond quat(current_pose_.rotation());
  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();
  
  // 发布odom
  nav_msgs::msg::Odometry odom_msg;
  odom_msg.header = header;
  odom_msg.header.frame_id = "map";
  odom_msg.child_frame_id = lidar_frame_;
  odom_msg.pose.pose = pose_msg.pose;
  odom_pub_->publish(odom_msg);

  // 更新path
  path_.header = header;
  path_.header.frame_id = "map";
  path_.poses.push_back(pose_msg);
  path_pub_->publish(path_);
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarOdometry>());
  rclcpp::shutdown();
  return 0;
}
