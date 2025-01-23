/**
 * @file common.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <execution>
#include <chrono>
#include <glog/logging.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "optimization_learning/so3_tool.hpp"

using H_b_type = std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>;

// bool next_iteration = false;
// void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
// {
//   if (event.getKeySym() == "space" && event.keyDown()) next_iteration = true;
// }


// pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
// viewer->setBackgroundColor(0, 0, 0);
// viewer->addPointCloud<pcl::PointXYZI>(source_points, "source_points");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_points");  // 红色
// viewer->addPointCloud<pcl::PointXYZI>(target_points, "target_points");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "target_points");  // 绿色
// viewer->addPointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "source_points_transformed");  // 蓝色

// viewer->spin();

// 逐步可视化
// viewer->registerKeyboardCallback (&keyboardEventOccurred, nullptr);
// while (!viewer->wasStopped ()) {
// viewer->spinOnce();
// if (next_iteration) {
//   // ...
//   viewer->updatePointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
// }
// next_iteration = false;

struct EIGEN_ALIGN16 _PointXYZIT {
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY
  float time;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 PointXYZIT : public _PointXYZIT
{
  // 构造函数
  inline PointXYZIT(const _PointXYZIT &p) : _PointXYZIT(p) {}
  
  inline PointXYZIT() 
  {
    x = y = z = 0.0f;
    data[3] = 1.0f;
    intensity = 0.0f;
    time = 0.0f;
  }

  inline PointXYZIT(float _x, float _y, float _z)
    : PointXYZIT()
  {
    x = _x; y = _y; z = _z;
  }

  inline PointXYZIT(float _x, float _y, float _z, float _intensity, float _time)
    : PointXYZIT(_x, _y, _z)
  {
    intensity = _intensity;
    time = _time;
  }

  friend std::ostream& operator<<(std::ostream& os, const PointXYZIT& p)
  {
    os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.intensity << "," << p.time << ")";
    return os;
  }
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

// 注册点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
)

POINT_CLOUD_REGISTER_POINT_WRAPPER(PointXYZIT, _PointXYZIT)

template class pcl::VoxelGrid<PointXYZIT>;
template class pcl::KdTreeFLANN<PointXYZIT>;
template class pcl::NormalEstimationOMP<PointXYZIT, pcl::Normal>;
struct EIGEN_ALIGN16 VelodynePoint {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
  VelodynePoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(std::uint16_t, ring, ring))

struct Pose6D {
  double offset_time;
  Eigen::Vector3d acc;
  Eigen::Vector3d gyr;
  Eigen::Vector3d vel;
  Eigen::Vector3d pos;
  Eigen::Quaterniond rot;
};

class TimeEval
{
public:
  TimeEval(const std::string& module_name = "unnamed") 
    : module_name_(module_name), is_timing_(false) {}
  
  ~TimeEval() {
    if (is_timing_) {
      LOG(WARNING) << "Warning: Timer '" << module_name_ << "' was not stopped properly";
    }
  }

  void tic() {
    if (is_timing_) {
      LOG(WARNING) << "Warning: Timer '" << module_name_ << "' was already started";
      return;
    }
    start_time_ = std::chrono::high_resolution_clock::now();
    is_timing_ = true;
  }

  void toc() {
    if (!is_timing_) {
      LOG(WARNING) << "Warning: Timer '" << module_name_ << "' was not started";
      return;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
    time_records_.emplace_back(duration.count() / 1000.0);  // 转换为毫秒
    is_timing_ = false;
  }

  struct TimingSummary {
    double avg_time;
    double max_time;
    double min_time;
    double std_dev;
    size_t count;
  };

  TimingSummary GetSummary() const {
    if (time_records_.empty()) {
      return {0.0, 0.0, 0.0, 0.0, 0};
    }

    double sum = std::accumulate(time_records_.begin(), time_records_.end(), 0.0);
    double avg = sum / time_records_.size();
    double max = *std::max_element(time_records_.begin(), time_records_.end());
    double min = *std::min_element(time_records_.begin(), time_records_.end());

    double sq_sum = std::accumulate(time_records_.begin(), time_records_.end(), 0.0,
      [avg](double acc, double val) {
        double diff = val - avg;
        return acc + diff * diff;
      });
    double std_dev = std::sqrt(sq_sum / time_records_.size());

    return {avg, max, min, std_dev, time_records_.size()};
  }

  void PrintSummary() const {
    auto summary = GetSummary();
    LOG(INFO) << "\nTiming Summary for '" << module_name_ << "':" << std::endl;
    LOG(INFO) << "  Count:     " << summary.count << std::endl;
    LOG(INFO) << "  Average:   " << std::fixed << std::setprecision(3) << summary.avg_time << " ms" << std::endl;
    LOG(INFO) << "  Maximum:   " << summary.max_time << " ms" << std::endl;
    LOG(INFO) << "  Minimum:   " << summary.min_time << " ms" << std::endl;
    LOG(INFO) << "  Std Dev:   " << summary.std_dev << " ms" << std::endl;
  }

  void ExportToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
      LOG(ERROR) << "Error: Could not open file " << filename;
      return;
    }

    // 写入CSV头
    file << "iteration,time_ms\n";

    // 写入每次的计时数据
    for (size_t i = 0; i < time_records_.size(); ++i) {
      file << i << "," << std::fixed << std::setprecision(3) << time_records_[i] << "\n";
    }

    // 写入统计数据
    auto summary = GetSummary();
    file << "\nSummary:\n";
    file << "count," << summary.count << "\n";
    file << "average," << summary.avg_time << "\n";
    file << "maximum," << summary.max_time << "\n";
    file << "minimum," << summary.min_time << "\n";
    file << "std_dev," << summary.std_dev << "\n";

    file.close();
  }

  std::string GetName() const {
    return module_name_;
  }

private:
  std::string module_name_;
  std::vector<double> time_records_;
  std::chrono::high_resolution_clock::time_point start_time_;
  bool is_timing_;
};