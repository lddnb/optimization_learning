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

  mean_acc_ = Eigen::Vector3d::Zero();
  mean_gyr_ = Eigen::Vector3d::Zero();
}

ImuIntegration::~ImuIntegration()
{
}

bool ImuIntegration::ProcessImu(
  const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer,
  const pcl::PointCloud<PointType>::Ptr& input_cloud,
  pcl::PointCloud<PointType>& output_cloud,
  std::unique_ptr<ESKF>& eskf)
{
  if (!is_initialized_) {
    Init(imu_buffer, eskf);
    return false;
  }

  UndistortPoint(imu_buffer, input_cloud, output_cloud, eskf);
  return true;
}

void ImuIntegration::Init(const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer, std::unique_ptr<ESKF>& eskf)
{
  static Eigen::Vector3d cur_acc = Eigen::Vector3d::Zero();
  static Eigen::Vector3d cur_gyr = Eigen::Vector3d::Zero();
  if (mean_acc_.isZero() && mean_gyr_.isZero()) {
    cur_acc << imu_buffer.front()->linear_acceleration.x, imu_buffer.front()->linear_acceleration.y,
      imu_buffer.front()->linear_acceleration.z;
    cur_gyr << imu_buffer.front()->angular_velocity.x, imu_buffer.front()->angular_velocity.y,
      imu_buffer.front()->angular_velocity.z;
    mean_acc_ = cur_acc;
    mean_gyr_ = cur_gyr;
    init_imu_size_++;
  }
  for (const auto& imu : imu_buffer) {
    cur_acc << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
    cur_gyr << imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z;
    mean_acc_ += (cur_acc - mean_acc_) / init_imu_size_;
    mean_gyr_ += (cur_gyr - mean_gyr_) / init_imu_size_;

    init_imu_size_++;
  }
  if (init_imu_size_ >= 100) {
    eskf->SetBiasAcc(mean_acc_);
    eskf->SetBiasGyro(mean_gyr_);
    
    const Eigen::Vector3d gravity_direction = -mean_acc_ / mean_acc_.norm();
    eskf->SetGravity(gravity_direction * 9.81);

    eskf->SetCovariance(Eigen::Matrix<double, 18, 18>::Identity() * 1e-4);

    const double ev = 1e-2;  // 加速度计测量标准差
    const double et = 1e-5;  // 陀螺仪测量标准差
    const double eg = 1e-6;  // 陀螺仪零偏随机游走标准差
    const double ea = 1e-4;  // 加速度计零偏随机游走标准差
    
    Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();
    Q.diagonal() << 0, 0, 0,  // 位置
                   ev, ev, ev,  // 速度
                   et, et, et,  // 姿态
                   eg, eg, eg,  // 陀螺仪偏置
                   ea, ea, ea,  // 加速度计偏置
                   0, 0, 0;     // 重力
    eskf->SetProcessNoise(Q);

    last_imu_msg_ = imu_buffer.back();
    is_initialized_ = true;
    
    LOG(INFO) << "IMU initialized with " << init_imu_size_ << " samples";
    LOG(INFO) << "Acceleration bias: " << eskf->GetBiasAcc().transpose();
    LOG(INFO) << "Gyroscope bias: " << eskf->GetBiasGyro().transpose();
    LOG(INFO) << "Gravity vector: " << eskf->GetGravity().transpose();
  }
}

void ImuIntegration::UndistortPoint(
  const std::deque<sensor_msgs::msg::Imu::SharedPtr>& imu_buffer,
  const pcl::PointCloud<PointType>::Ptr& point_cloud,
  pcl::PointCloud<PointType>& undistorted_point_cloud,
  std::unique_ptr<ESKF>& eskf)
{
  static Eigen::Vector3d acc_s_last = Eigen::Vector3d::Zero();
  static Eigen::Vector3d angvel_last = Eigen::Vector3d::Zero();
  static std::vector<Pose6D> imu_pose_list;
  static double last_lidar_end_time_ = 0.0;

  auto imu_msgs = imu_buffer;
  imu_msgs.push_front(last_imu_msg_);
  const double imu_begin_timestamp = imu_msgs.front()->header.stamp.sec + imu_msgs.front()->header.stamp.nanosec * 1e-9;
  const double imu_end_timestamp = imu_msgs.back()->header.stamp.sec + imu_msgs.back()->header.stamp.nanosec * 1e-9;

  undistorted_point_cloud = *point_cloud;
  std::sort(std::execution::par, undistorted_point_cloud.points.begin(), undistorted_point_cloud.points.end(), time_list<PointType>);
  const double point_begin_timestamp = undistorted_point_cloud.points.front().time;
  const double point_end_timestamp = undistorted_point_cloud.points.back().time;

  imu_pose_list.clear();

  Pose6D last_frame_final_point_pose;
  last_frame_final_point_pose.offset_time = 0;
  last_frame_final_point_pose.acc = acc_s_last;
  last_frame_final_point_pose.gyr = angvel_last;
  last_frame_final_point_pose.vel = eskf->GetVelocity();
  last_frame_final_point_pose.pos = eskf->GetPosition();
  last_frame_final_point_pose.rot = eskf->GetRotation();

  imu_pose_list.emplace_back(last_frame_final_point_pose);

  Eigen::Vector3d angvel_avr, acc_avr, vel_imu, pos_imu;
  Eigen::Quaterniond R_imu;

  double dt = 0;

  // 这里相当于第一个imu数据的时刻是没有去算状态的，因为这是上一帧的最后一个imu数据
  // 而上一帧最后一个点云数据的时间是比第一个imu数据的时间要晚的
  // 所以直接在last_frame_final_point_pose的基础上进行状态传播
  // 后面的每一个 imu 数据时刻的状态量都进行了计算
  for (auto it_imu = imu_msgs.begin(); it_imu != std::prev(imu_msgs.end()); ++it_imu)
  {
    const auto head = *(it_imu);
    const auto tail = *(std::next(it_imu));

    const double head_time = head->header.stamp.sec + head->header.stamp.nanosec * 1e-9;
    const double tail_time = tail->header.stamp.sec + tail->header.stamp.nanosec * 1e-9;
    
    if (tail_time < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
      0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
      0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
      0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
      0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr = acc_avr * 9.81 / mean_acc_.norm();

    if (head_time < last_lidar_end_time_) {
      dt = tail_time - last_lidar_end_time_;
    } else {
      dt = tail_time - head_time;
    }

    eskf->Predict(acc_avr, angvel_avr, dt);

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr - eskf->GetBiasGyro();
    acc_s_last  = eskf->GetRotation() * (acc_avr - eskf->GetBiasAcc());
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += eskf->GetGravity()[i];
    }
    double &&offs_t = tail_time - point_begin_timestamp;
    Pose6D imu_pose;
    imu_pose.offset_time = offs_t;
    imu_pose.acc = acc_s_last;
    imu_pose.gyr = angvel_last;
    imu_pose.vel = eskf->GetVelocity();
    imu_pose.pos = eskf->GetPosition();
    imu_pose.rot = eskf->GetRotation();
    imu_pose_list.emplace_back(imu_pose);
  }

  const double note = point_end_timestamp > imu_end_timestamp ? 1.0 : -1.0;
  dt = note * (point_end_timestamp - imu_end_timestamp);
  eskf->Predict(acc_avr, angvel_avr, dt);


  auto final_point_pos = eskf->GetPosition();
  auto final_point_rot = eskf->GetRotation();
  last_imu_msg_ = imu_msgs.back();
  last_lidar_end_time_ = point_end_timestamp;

  auto it_point = undistorted_point_cloud.points.rbegin();
  for (auto it_imu = imu_pose_list.crbegin(); it_imu < imu_pose_list.crend(); ++it_imu) {
    const auto tail = *it_imu;
    const auto head = *(std::next(it_imu));

    pos_imu = head.pos;
    vel_imu = head.vel;
    R_imu = head.rot;
    acc_avr = tail.acc;
    angvel_avr = tail.gyr;
    LOG_FIRST_N(INFO, 100) << "head.offset_time: " << head.offset_time << " it_point->time: " << it_point->time;
    LOG_FIRST_N(INFO, 100) << "pos_imu: " << head.pos.transpose();
    LOG_FIRST_N(INFO, 100) << "vel_imu: " << head.vel.transpose();
    LOG_FIRST_N(INFO, 100) << "R_imu: " << head.rot.matrix();
    LOG_FIRST_N(INFO, 100) << "acc_avr: " << acc_avr.transpose();
    LOG_FIRST_N(INFO, 100) << "angvel_avr: " << angvel_avr.transpose();

    for (; it_point->time > head.offset_time; ++it_point) {
      if (it_point == undistorted_point_cloud.points.rend()) break;

      double dt = it_point->time - head.offset_time;
      Eigen::Matrix3d R_i = R_imu * Exp(angvel_avr * dt);
      Eigen::Vector3d pt_i = it_point->getVector3fMap().cast<double>();
      Eigen::Vector3d t_ei = pos_imu + vel_imu * dt + 0.5 * acc_avr * dt * dt - final_point_pos;
      Eigen::Vector3d pt_i_new = final_point_rot.inverse() * (R_i * pt_i + t_ei);
      it_point->getVector3fMap() = pt_i_new.cast<float>();
      LOG_FIRST_N(INFO, 100) << "pt_i: " << pt_i.transpose() << " pt_i_new: " << pt_i_new.transpose();
    }
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

int ImuIntegration::GetInitImuSize() const
{
  return init_imu_size_;
}