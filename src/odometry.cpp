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
#include "optimization_learning/downsampling.hpp"

LidarOdometry::LidarOdometry() : Node("lidar_odometry")
{
  LOG(INFO) << "LidarOdometry Initializing";

  ConfigRegistration();

  // 设置QoS
  rclcpp::SensorDataQoS qos;
  qos.reliable();
  qos.keep_last(100);

  // 创建订阅者和发布者
  const auto lidar_topic = this->declare_parameter("lidar_topic", std::string());
  const auto imu_topic = this->declare_parameter("imu_topic", std::string());
  const auto ground_truth_path_topic = this->declare_parameter("ground_truth_path_topic", std::string());
  const auto odom_topic = this->declare_parameter("output_odom_topic", std::string());
  const auto path_topic = this->declare_parameter("output_path_topic", std::string());
  const auto cloud_topic = this->declare_parameter("output_cloud_topic", std::string());
  LOG(INFO) << "[cfg] lidar_topic: " << lidar_topic;
  LOG(INFO) << "[cfg] imu_topic: " << imu_topic;
  LOG(INFO) << "[cfg] ground_truth_path_topic: " << ground_truth_path_topic;

  local_map_min_frame_size_ = this->declare_parameter("local_map_min_frame_size", int(0));
  update_frame_size_ = this->declare_parameter("update_frame_size", int(0));
  update_translation_delta_ = this->declare_parameter("update_translation_delta", double(0.0));
  update_rotation_delta_ = this->declare_parameter("update_rotation_delta", double(0.0));
  LOG(INFO) << "[cfg] local_map_min_frame_size: " << local_map_min_frame_size_;
  LOG(INFO) << "[cfg] update_frame_size: " << update_frame_size_;
  LOG(INFO) << "[cfg] update_translation_delta: " << update_translation_delta_;
  LOG(INFO) << "[cfg] update_rotation_delta: " << update_rotation_delta_;
  update_rotation_delta_ = update_rotation_delta_ * M_PI / 180;

  save_map_path_ = this->declare_parameter("save_map_path", std::string());
  LOG(INFO) << "[cfg] save_map_path: " << save_map_path_;

  auto T_imu2lidar = this->declare_parameter("T_imu2lidar", std::vector<double>());
  T_imu2lidar_ = Eigen::Isometry3d::Identity();
  T_imu2lidar_.matrix() << T_imu2lidar[0], T_imu2lidar[1], T_imu2lidar[2], T_imu2lidar[3],
                        T_imu2lidar[4], T_imu2lidar[5], T_imu2lidar[6], T_imu2lidar[7],
                        T_imu2lidar[8], T_imu2lidar[9], T_imu2lidar[10], T_imu2lidar[11],
                        T_imu2lidar[12], T_imu2lidar[13], T_imu2lidar[14], T_imu2lidar[15];
  LOG(INFO) << "[cfg] T_imu2lidar: R:" << T_imu2lidar_.linear() << std::endl << "t:" << T_imu2lidar_.translation().transpose();

  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    lidar_topic, qos,
    std::bind(&LidarOdometry::CloudCallback, this, std::placeholders::_1));
  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, qos,
    std::bind(&LidarOdometry::ImuCallback, this, std::placeholders::_1));
  ground_truth_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
    ground_truth_path_topic, qos,
    std::bind(&LidarOdometry::GroundTruthPathCallback, this, std::placeholders::_1));
  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
  path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic, 10);
  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic, 10);
  
  current_pose_ = Eigen::Isometry3d::Identity();
  LOG(INFO) << "LidarOdometry Initialized";

  // 初始化tf广播器
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  imu_integration_ = std::make_unique<ImuIntegration>();
  imu_integration_->SetBias(Eigen::Vector3d(0.0001, 0.0001, 0.0001), Eigen::Vector3d(0.0001, 0.0001, 0.0001));

  eskf_ = std::make_unique<ESKF>();

  // 初始化时间评估
  downsample_time_eval_ = TimeEval("downsample");
  registration_time_eval_ = TimeEval("registration");
  
  main_thread_ = std::make_unique<std::thread>(&LidarOdometry::MainThread, this);
}

LidarOdometry::~LidarOdometry()
{
  SaveMappingResult();
  if (main_thread_ && main_thread_->joinable()) {
    main_thread_->join();
  }
  main_thread_.reset();
}

void LidarOdometry::MainThread()
{
  static sensor_msgs::msg::PointCloud2::SharedPtr current_cloud_buffer = nullptr;
  static std::deque<sensor_msgs::msg::Imu::SharedPtr> current_imu_buffer;
  while (rclcpp::ok()) {
    if (GetSyncedData(current_cloud_buffer, current_imu_buffer)) {
      LOG_FIRST_N(INFO, 10) << "Get synced data, imu size: " << current_imu_buffer.size();
      auto start_time = std::chrono::high_resolution_clock::now();
      pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
      // pcl::fromROSMsg(*current_cloud_buffer, *cloud);
      VelodyneHandler(current_cloud_buffer, *cloud);

      // 下采样
      pcl::PointCloud<PointType>::Ptr downsampled_cloud(new pcl::PointCloud<PointType>);
      downsample_time_eval_.tic();
      // voxel_grid_.setInputCloud(cloud);
      // voxel_grid_.filter(*downsampled_cloud);
      downsampled_cloud = voxelgrid_sampling_pstl<PointType>(cloud, 0.4);
      downsample_time_eval_.toc();

      // 转换到imu坐标系
      pcl::transformPointCloud(*downsampled_cloud, *downsampled_cloud, T_imu2lidar_.matrix());

      static int cnt = 0;
      pcl::PointCloud<PointType>::Ptr undistorted_cloud(new pcl::PointCloud<PointType>(*downsampled_cloud));
      if (!imu_integration_->ProcessImu(current_imu_buffer, downsampled_cloud, *undistorted_cloud, eskf_)) {
        LOG(WARNING) << "IMU initializing " << imu_integration_->GetInitImuSize() << " frames";
        continue;
      }
      if (cnt < 10) pcl::io::savePCDFileASCII("/home/ubuntu/data/NCLT/before_undistorted_cloud_" + std::to_string(cnt) + ".pcd", *downsampled_cloud);
      if (cnt < 10) pcl::io::savePCDFileASCII("/home/ubuntu/data/NCLT/after_undistorted_cloud_" + std::to_string(cnt) + ".pcd", *undistorted_cloud);
      cnt++;

      LOG(INFO) << "undistorted_cloud size: " << undistorted_cloud->size();

      if (local_map_ == nullptr) {
        local_map_.reset(new pcl::PointCloud<PointType>);
        local_map_ = undistorted_cloud;
        pcl::io::savePCDFileASCII("/home/ubuntu/data/NCLT/local_map_0.pcd", *local_map_);
        PublishTF(current_cloud_buffer->header);
        PublishOdom(current_cloud_buffer->header);
        continue;
      }
      LOG(INFO) << "local_map_ size: " << local_map_->size();
      static bool save_local_map = false;
      if (!save_local_map) {
        pcl::io::savePCDFileASCII("/home/ubuntu/data/NCLT/local_map_1.pcd", *local_map_);
        save_local_map = true;
      }

      // LOG_FIRST_N(INFO, 10000) << "pose before: " << current_pose_.translation().transpose();
      // imu_integration_->UpdatePose(current_pose_);
      // for (const auto& imu : current_imu_buffer) {
      //   imu_integration_->Integrate(imu);
      // }
      // current_pose_ = imu_integration_->GetCurrentPose();
      // LOG_FIRST_N(INFO, 10000) << "pose after: " << current_pose_.translation().transpose();
      // LOG_FIRST_N(INFO, 10000) << "velocity: " << imu_integration_->GetCurrentVelocity().transpose();

      registration_time_eval_.tic();
      registration->setInputSource(undistorted_cloud);
      registration->setInputTarget(local_map_);
      registration->setInitialTransformation(current_pose_);
      int iterations = 0;
      registration->align(current_pose_, iterations);
      registration_time_eval_.toc();

      // 发布pose和tf
      PublishTF(current_cloud_buffer->header);
      PublishOdom(current_cloud_buffer->header);

      pcl::transformPointCloud(*undistorted_cloud, *undistorted_cloud, current_pose_.matrix());
      sensor_msgs::msg::PointCloud2 cloud_msg;
      pcl::toROSMsg(*undistorted_cloud, cloud_msg);
      //! 记得设置 frame_id
      cloud_msg.header.frame_id = "map";
      cloud_msg.header.stamp = current_cloud_buffer->header.stamp;
      cloud_pub_->publish(cloud_msg);

      UpdateLocalMap(undistorted_cloud, current_pose_);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      LOG(INFO) << "Elapsed time: " << duration << " ms, iterations: " << iterations;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

bool LidarOdometry::GetSyncedData(sensor_msgs::msg::PointCloud2::SharedPtr& cloud_buffer, std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer)
{
  if (lidar_cloud_buffer_.size() < 2 || imu_buffer_.size() < 5) {
    return false;
  }
  cloud_buffer = nullptr;
  imu_buffer.clear();
  // LOG(INFO) << "Get synced data, lidar size: " << lidar_cloud_buffer_.size() << ", imu size: " << imu_buffer_.size();
  double lidar_frame_begin_time = lidar_cloud_buffer_.front()->header.stamp.sec + lidar_cloud_buffer_.front()->header.stamp.nanosec * 1e-9;
  double lidar_frame_end_time = lidar_cloud_buffer_[1]->header.stamp.sec + lidar_cloud_buffer_[1]->header.stamp.nanosec * 1e-9;
  double imu_begin_time = imu_buffer_.front()->header.stamp.sec + imu_buffer_.front()->header.stamp.nanosec * 1e-9;
  double imu_end_time = imu_buffer_.back()->header.stamp.sec + imu_buffer_.back()->header.stamp.nanosec * 1e-9;

  // 三种情况
  // 1. lidar_frame_begin_time > imu_end_time，清空imu
  // 2. lidar_frame_end_time < imu_begin_time，清空lidar
  // 3. lidar_frame_begin_time <= imu_begin_time && lidar_frame_end_time >= imu_end_time，同步，取点云扫描时间内的imu

  if (lidar_frame_begin_time > imu_end_time) {
    std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
    imu_buffer_.clear();
    return false;
  }
  if (lidar_frame_end_time < imu_begin_time) {
    std::lock_guard<std::mutex> lock(cloud_buffer_mutex_);
    lidar_cloud_buffer_.pop_front();
    return false;
  }

  // 同步
  {
    std::lock_guard<std::mutex> lock(cloud_buffer_mutex_);
    cloud_buffer = lidar_cloud_buffer_.front();
  }
  {
    std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
    for (const auto& imu : imu_buffer_) {
      if (imu->header.stamp.sec + imu->header.stamp.nanosec * 1e-9 > lidar_frame_begin_time &&
          imu->header.stamp.sec + imu->header.stamp.nanosec * 1e-9 < lidar_frame_end_time) {
        imu_buffer.emplace_back(imu);
        imu_buffer_.pop_front();
      } else if (imu->header.stamp.sec + imu->header.stamp.nanosec * 1e-9 > lidar_frame_end_time) {
        break;
      }
    }
  }
  if (imu_buffer.size() < 5) {
    return false;
  }

  return true;
}

RegistrationConfig LidarOdometry::ConfigRegistration()
{
  // 读取参数
  RegistrationConfig reg_config;
  reg_config.registration_type = static_cast<RegistrationConfig::RegistrationType>(
    this->declare_parameter("registration_type", int(0)));
  reg_config.solve_type = static_cast<RegistrationConfig::SolveType>(
    this->declare_parameter("solve_type", int(0)));
  reg_config.max_correspondence_distance = this->declare_parameter("max_correspondence_distance", double(0.0));
  reg_config.max_iterations = this->declare_parameter("max_iterations", int(0));
  reg_config.translation_eps = this->declare_parameter("translation_epsilon", double(0.0));
  reg_config.rotation_eps = this->declare_parameter("rotation_epsilon", double(0.0));
  reg_config.resolution = this->declare_parameter("resolution", double(0.0));
  reg_config.num_threads = this->declare_parameter("num_threads", int(0));
  reg_config.verbose = this->declare_parameter("verbose", bool(false));
  reg_config.downsample_leaf_size = this->declare_parameter("downsample_leaf_size", double(0.0));

  // 创建配准对象
  switch (reg_config.registration_type) {
    case RegistrationConfig::ICP: {
      registration = std::make_unique<ICPRegistration<PointType>>(reg_config);
      break;
    }
    case RegistrationConfig::NICP: {
      registration = std::make_unique<NICPRegistration<PointType>>(reg_config);
      break;
    }
    case RegistrationConfig::GICP: {
      registration = std::make_unique<GICPRegistration<PointType>>(reg_config);
      break;
    }
    case RegistrationConfig::NDT: {
      registration = std::make_unique<NDTRegistration<PointType>>(reg_config);
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

void LidarOdometry::SaveMappingResult()
{
  // 保存TUM格式
  std::ofstream file(save_map_path_ + "/gt_trajectory.txt");
  for (const auto& pose : ground_truth_path_.poses) {
    std::string timestamp = std::to_string(pose.header.stamp.sec) + "." + std::to_string(pose.header.stamp.nanosec);
    file << timestamp << " " << pose.pose.position.x << " " << pose.pose.position.y << " " << pose.pose.position.z << " " << pose.pose.orientation.x << " " << pose.pose.orientation.y << " " << pose.pose.orientation.z << " " << pose.pose.orientation.w << std::endl;
  }
  file.close();

  file.open(save_map_path_ + "/mapping_trajectory.txt");
  for (const auto& pose : path_.poses) {
    std::string timestamp = std::to_string(pose.header.stamp.sec) + "." + std::to_string(pose.header.stamp.nanosec);
    file << timestamp << " " << pose.pose.position.x << " " << pose.pose.position.y << " " << pose.pose.position.z << " " << pose.pose.orientation.x << " " << pose.pose.orientation.y << " " << pose.pose.orientation.z << " " << pose.pose.orientation.w << std::endl;
  }
  file.close();
  LOG(INFO) << "Save mapping result to " << save_map_path_;

  std::string downsample_time_eval_path = save_map_path_ + "/" + downsample_time_eval_.GetName() + ".csv";
  downsample_time_eval_.ExportToFile(downsample_time_eval_path);

  std::string registration_time_eval_path = save_map_path_ + "/" + registration_time_eval_.GetName() + ".csv";
  registration_time_eval_.ExportToFile(registration_time_eval_path);
}

void LidarOdometry::CloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  static int lidar_frame_index = 0;
  double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  LOG_FIRST_N(INFO, 10) << "stamp: " << stamp;
  
  // 将UTC时间戳转换为时间结构
  time_t time_stamp = static_cast<time_t>(stamp);
  struct tm* timeinfo = localtime(&time_stamp);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
  
  LOG_FIRST_N(INFO, 10) << "lidar_frame_index: " << lidar_frame_index++ 
            << " with stamp: " << buffer 
            << "." << std::setfill('0') << std::setw(9) << msg->header.stamp.nanosec;

  if (lidar_frame_.empty()) {
    lidar_frame_ = msg->header.frame_id;
  }

  {
    std::lock_guard<std::mutex> lock(cloud_buffer_mutex_);
    // 添加缓冲区大小限制
    if (lidar_cloud_buffer_.size() > 1000) {
      LOG(WARNING) << "Buffer overflow, dropping oldest data";
      lidar_cloud_buffer_.pop_front();
    }
    lidar_cloud_buffer_.emplace_back(msg);
  }
}

void LidarOdometry::ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
  imu_buffer_.emplace_back(msg);

  if (imu_buffer_.size() > 20000) {
    LOG(WARNING) << "Buffer overflow, dropping oldest data";
    imu_buffer_.pop_front();
  }
}

void LidarOdometry::GroundTruthPathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
  ground_truth_path_ = *msg;
}

void LidarOdometry::UpdateLocalMap(const pcl::PointCloud<PointType>::Ptr& msg, const Eigen::Isometry3d& pose)
{
  static std::deque<pcl::PointCloud<PointType>::Ptr> local_map_list;
  static Eigen::Isometry3d last_pose = Eigen::Isometry3d::Identity();
  auto delta_pose = pose * last_pose.inverse();
  last_pose = pose;
  if (local_map_list.empty() || delta_pose.translation().norm() > update_translation_delta_ ||
      Eigen::AngleAxisd(delta_pose.rotation()).angle() > update_rotation_delta_) {
    local_map_list.push_back(msg);
    *local_map_ += *msg;
  }

  if (
    local_map_list.size() >= local_map_min_frame_size_ &&
    (local_map_list.size() >= update_frame_size_)) {
    local_map_.reset(new pcl::PointCloud<PointType>);
    for (const auto& cloud : local_map_list) {
      *local_map_ += *cloud;
    }
    LOG(INFO) << "Update local map size: " << local_map_list.size();
    local_map_list.clear();
  }
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
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  // google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarOdometry>());
  rclcpp::shutdown();
  return 0;
}
