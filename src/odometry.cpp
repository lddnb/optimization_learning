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

#define USE_ESKF 1

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
  const auto local_map_topic = this->declare_parameter("output_local_map_topic", std::string());
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
    lidar_topic,
    qos,
    std::bind(&LidarOdometry::CloudCallback, this, std::placeholders::_1));
  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic,
    qos,
    std::bind(&LidarOdometry::ImuCallback, this, std::placeholders::_1));
  ground_truth_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
    ground_truth_path_topic,
    qos,
    std::bind(&LidarOdometry::GroundTruthPathCallback, this, std::placeholders::_1));
  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
  path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic, 10);
  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic, 10);
  local_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(local_map_topic, 10);

  current_pose_ = Eigen::Isometry3d::Identity();

  // 初始化tf广播器
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

  const int imu_init_sec = this->declare_parameter("imu_init_sec", int(0));
  imu_integration_ = std::make_unique<ImuIntegration>(imu_init_sec);
  imu_integration_->SetBias(Eigen::Vector3d(0.0001, 0.0001, 0.0001), Eigen::Vector3d(0.0001, 0.0001, 0.0001));

  eskf_ = std::make_unique<ESKF>();

  // 初始化时间评估
  downsample_time_eval_ = TimeEval("downsample");
  registration_time_eval_ = TimeEval("registration");

  // local map
  local_map_buffer_.reset(new pcl::PointCloud<PointType>);
  local_map_buffer_size_ = 0;

  synced_data_.cloud.reset(new pcl::PointCloud<PointType>);
  
  main_thread_ = std::make_unique<std::thread>(&LidarOdometry::MainThread, this);

  LOG(INFO) << "LidarOdometry Initialized";
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
  while (rclcpp::ok()) {
    if (GetSyncedData(synced_data_)) {
      LOG_FIRST_N(INFO, 10) << "Get synced data, imu size: " << synced_data_.imu.size();
      auto start_time = std::chrono::high_resolution_clock::now();

      builtin_interfaces::msg::Time ros_stamp;
      ros_stamp.sec = static_cast<int>(synced_data_.lidar_begin_timestamp);
      ros_stamp.nanosec = static_cast<int>((synced_data_.lidar_begin_timestamp - static_cast<int>(synced_data_.lidar_begin_timestamp)) * 1e9);

      // 预测步 + 去畸变
      pcl::PointCloud<PointType>::Ptr undistorted_cloud(new pcl::PointCloud<PointType>(*synced_data_.cloud));
#if USE_ESKF
      if (!imu_integration_->ProcessImu(synced_data_, *undistorted_cloud, eskf_)) {
        LOG(WARNING) << "IMU initializing " << imu_integration_->GetInitImuSize() << " frames";
        continue;
      }
#endif

      // 下采样
      pcl::PointCloud<PointType>::Ptr downsampled_cloud(new pcl::PointCloud<PointType>);
      downsample_time_eval_.tic();
      // voxel_grid_.setInputCloud(cloud);
      // voxel_grid_.filter(*downsampled_cloud);
      downsampled_cloud = voxelgrid_sampling_pstl<PointType>(undistorted_cloud, 0.4);
      downsample_time_eval_.toc();

      // 转换到imu坐标系
      pcl::transformPointCloud(*downsampled_cloud, *downsampled_cloud, T_imu2lidar_.matrix());

      // local map初始化
      if (local_map_ == nullptr) {
        local_map_.reset(new pcl::PointCloud<PointType>);
        local_map_ = downsampled_cloud;
        ikdtree_.set_downsample_param(0.5);
        ikdtree_.Build(downsampled_cloud->points);
        PublishTF(ros_stamp);
        PublishOdom(ros_stamp);
        continue;
      }

#if USE_ESKF
      current_pose_.translation() = eskf_->GetPosition();
      current_pose_.linear() = eskf_->GetRotation().toRotationMatrix();
#endif

      // 点云配准
      registration_time_eval_.tic();
      registration->setInputSource(downsampled_cloud);
      registration->setInputTarget(local_map_);
      registration->setInitialTransformation(current_pose_);
      int iterations = 0;
      registration->align(current_pose_, iterations);
      registration_time_eval_.toc();

      LOG(INFO) << "current_pose_ after registration: " << current_pose_.translation().transpose();

#if USE_ESKF
      // eskf 更新步
      eskf_->Update(current_pose_.translation(), Eigen::Quaterniond(current_pose_.rotation()), 1e-2, 1e-2);
      current_pose_.translation() = eskf_->GetPosition();
      current_pose_.linear() = eskf_->GetRotation().toRotationMatrix();
      LOG(INFO) << "current_pose_ after eskf: " << current_pose_.translation().transpose();
#endif

      // 发布pose和tf
      PublishTF(ros_stamp);
      PublishOdom(ros_stamp);

      // 发布点云
      pcl::transformPointCloud(*downsampled_cloud, *downsampled_cloud, current_pose_.matrix());
      sensor_msgs::msg::PointCloud2 cloud_msg;
      pcl::toROSMsg(*downsampled_cloud, cloud_msg);
      //! 记得设置 frame_id
      cloud_msg.header.frame_id = "map";
      cloud_msg.header.stamp = ros_stamp;
      cloud_pub_->publish(cloud_msg);

      // 更新 local map
      UpdateLocalMap(downsampled_cloud, current_pose_);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      LOG(INFO) << "Elapsed time: " << duration << " ms, iterations: " << iterations;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

bool LidarOdometry::GetSyncedData(SyncedData& synced_data)
{
  if (lidar_cloud_buffer_.size() < 2 || imu_buffer_.size() < 5) {
    return false;
  }
  
  synced_data.cloud->clear();
  synced_data.imu.clear();
  const double lidar_begin_time = GetTimestamp(lidar_cloud_buffer_.front());
  const double lidar_end_time = GetTimestamp(lidar_cloud_buffer_[1]);
  const double imu_begin_time = GetTimestamp(imu_buffer_.front());
  const double imu_end_time = GetTimestamp(imu_buffer_.back());
  
  // 三种情况
  // 1. lidar_frame_begin_time > imu_end_time，清空imu
  // 2. lidar_frame_end_time < imu_begin_time，清空lidar
  // 3. lidar_frame_begin_time <= imu_begin_time && lidar_frame_end_time >= imu_end_time，同步，取点云扫描时间内的imu

  if (lidar_begin_time > imu_end_time) {
    std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
    imu_buffer_.clear();
    return false;
  }
  if (lidar_end_time < imu_begin_time) {
    std::lock_guard<std::mutex> lock(cloud_buffer_mutex_);
    lidar_cloud_buffer_.pop_front();
    return false;
  }

  // 同步
  {
    std::lock_guard<std::mutex> lock(cloud_buffer_mutex_);
    // pcl::fromROSMsg(*lidar_cloud_buffer_.front(), *synced_data_.cloud);
    VelodyneHandler(lidar_cloud_buffer_.front(), *synced_data_.cloud);
    lidar_cloud_buffer_.pop_front();
  }
  {
    std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
    for (const auto& imu : imu_buffer_) {
      if (imu->header.stamp.sec + imu->header.stamp.nanosec * 1e-9 > lidar_begin_time &&
          imu->header.stamp.sec + imu->header.stamp.nanosec * 1e-9 < lidar_end_time) {
        synced_data.imu.emplace_back(imu);
        imu_buffer_.pop_front();
      } else if (imu->header.stamp.sec + imu->header.stamp.nanosec * 1e-9 > lidar_end_time) {
        break;
      }
    }
  }
  if (synced_data.imu.size() < 5) {
    return false;
  }

  synced_data.lidar_begin_timestamp = lidar_begin_time;
  synced_data.lidar_end_timestamp = lidar_end_time;

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

template <typename MsgPtr>
double LidarOdometry::GetTimestamp(const MsgPtr& msg)
{
  return msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
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
  static Eigen::Isometry3d last_pose = pose;

  // 计算相对运动（在last_pose局部坐标系下）
  const Eigen::Vector3d delta_t = pose.translation() - last_pose.translation();
  const Eigen::Matrix3d delta_R = last_pose.rotation().transpose() * pose.rotation();
  const double trans_diff = delta_t.norm();
  const double rot_diff = Eigen::AngleAxisd(delta_R).angle();

  if (local_map_buffer_->empty() || trans_diff > update_translation_delta_ || rot_diff > update_rotation_delta_) {
    *local_map_buffer_ += *msg;
    *local_map_ += *msg;
    last_pose = pose;
    local_map_buffer_size_++;
    // LOG(INFO) << "Position change: " << trans_diff << "m, " << "Rotation change: " << (rot_diff * 180.0 / M_PI) << "deg";
  }

  if (local_map_buffer_size_ >= update_frame_size_) {
    local_map_->clear();
    // *local_map_ = *local_map_buffer_;
    local_map_ = voxelgrid_sampling_pstl<PointType>(local_map_buffer_, 0.2);
    LOG(WARNING) << "Update local map! point size: " << local_map_->size();
    local_map_buffer_->clear();
    local_map_buffer_size_ = 0;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*local_map_, cloud_msg);
    //! 记得设置 frame_id
    cloud_msg.header.frame_id = "map";
    cloud_msg.header.stamp = now();
    local_map_pub_->publish(cloud_msg);
  }
}

void LidarOdometry::PublishTF(const builtin_interfaces::msg::Time& timestamp)
{
  geometry_msgs::msg::TransformStamped transform_stamped;
  
  // 设置时间戳稍早于点云消息
  transform_stamped.header.stamp = timestamp;
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

void LidarOdometry::PublishOdom(const builtin_interfaces::msg::Time& timestamp)
{
  // 发布pose
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.stamp = timestamp;
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
  odom_msg.header.stamp = timestamp;
  odom_msg.header.frame_id = "map";
  odom_msg.child_frame_id = lidar_frame_;
  odom_msg.pose.pose = pose_msg.pose;

  const auto& P = eskf_->GetCovariance();
  
  // 使用 Eigen 的块操作
  Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> covar(odom_msg.pose.covariance.data());
  covar.block<3,3>(0,0) = P.block<3,3>(0,0);    // 位置协方差
  covar.block<3,3>(0,3) = P.block<3,3>(0,6);    // 位置-姿态协方差
  covar.block<3,3>(3,0) = P.block<3,3>(6,0);    // 姿态-位置协方差
  covar.block<3,3>(3,3) = P.block<3,3>(6,6);    // 姿态协方差

  odom_pub_->publish(odom_msg);

  // 更新path
  path_.header.stamp = timestamp;
  path_.header.frame_id = "map";
  path_.poses.emplace_back(pose_msg);
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
