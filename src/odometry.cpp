
#include "optimization_learning/odometry.hpp"

LidarOdometry::LidarOdometry(const RegistrationConfig& config) : Node("lidar_odometry")
{
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "input_cloud",
    10,
    std::bind(&LidarOdometry::cloudCallback, this, std::placeholders::_1));
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("output_pose", 10);

  switch (config.registration_type) {
    case RegistrationConfig::ICP: {
      registration = std::make_unique<ICPRegistration<pcl::PointXYZI>>(config);
      break;
    }
    case RegistrationConfig::NICP: {
      registration = std::make_unique<NICPRegistration<pcl::PointXYZI>>(config);
      break;
    }
    case RegistrationConfig::GICP: {
      registration = std::make_unique<GICPRegistration<pcl::PointXYZI>>(config);
      break;
    }
    case RegistrationConfig::NDT: {
      registration = std::make_unique<NDTRegistration<pcl::PointXYZI>>(config);
      break;
    }
    default: {
      LOG(ERROR) << "Unknown registration type";
      return;
    }
  }

  current_pose_ = Eigen::Isometry3d::Identity();
}

LidarOdometry::~LidarOdometry() {}

void LidarOdometry::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *cloud);

  if (previous_cloud_ == nullptr) {
    previous_cloud_ = cloud;
    return;
  }

  registration->setInputSource(cloud);
  registration->setInputTarget(previous_cloud_);
  registration->setInitialTransformation(current_pose_);
  int iterations = 0;
  registration->align(current_pose_, iterations);

  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header = msg->header;
  pose_msg.pose.position.x = current_pose_.translation().x();
  pose_msg.pose.position.y = current_pose_.translation().y();
  pose_msg.pose.position.z = current_pose_.translation().z();
  Eigen::Matrix3d rotation = current_pose_.linear().matrix();
  Eigen::Quaterniond quat(rotation);
  pose_msg.pose.orientation.x = quat.x();
  pose_msg.pose.orientation.y = quat.y();
  pose_msg.pose.orientation.z = quat.z();
  pose_msg.pose.orientation.w = quat.w();

  pose_pub_->publish(pose_msg);

  previous_cloud_ = cloud;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  RegistrationConfig config;
  rclcpp::spin(std::make_shared<LidarOdometry>(config));
  rclcpp::shutdown();
  return 0;
}
