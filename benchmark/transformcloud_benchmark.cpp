/**
 * @file transformcloud_benchmark.cpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <benchmark/benchmark.h>
#include <execution>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "optimization_learning/transform_point_cloud.hpp"

class TransformCloud : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State& state) {
    std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path + "target.pcd", *cloud);

    LOG_FIRST_N(INFO, 1) << "cloud size: " << cloud->size();

    // 创建一个随机变换矩阵
    Eigen::AngleAxisf rotation_vector(M_PI / 4, Eigen::Vector3f(1, 0, 0));  // 绕X轴旋转45度
    transform.linear() = rotation_vector.toRotationMatrix();
    transform.translation() = Eigen::Vector3f(1.0, 2.0, 3.0);  // 平移(1,2,3)
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
};

// Benchmark our implementation
BENCHMARK_DEFINE_F(TransformCloud, CustomTransform)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudPSTL(*cloud, *transformed_cloud, transform);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

// Benchmark PCL's implementation
BENCHMARK_DEFINE_F(TransformCloud, PCLTransform)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform.matrix());
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

// Benchmark sequential implementation
BENCHMARK_DEFINE_F(TransformCloud, SequentialTransform)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudSEQ(*cloud, *transformed_cloud, transform);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

// 在TransformCloud类中添加测试用例
BENCHMARK_DEFINE_F(TransformCloud, CustomTransformOMP_1Thread)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudOMP(*cloud, *transformed_cloud, transform, 1);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

BENCHMARK_DEFINE_F(TransformCloud, CustomTransformOMP_2Thread)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudOMP(*cloud, *transformed_cloud, transform, 2);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

BENCHMARK_DEFINE_F(TransformCloud, CustomTransformOMP_4Thread)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudOMP(*cloud, *transformed_cloud, transform, 4);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

BENCHMARK_DEFINE_F(TransformCloud, CustomTransformOMP_8Thread)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudOMP(*cloud, *transformed_cloud, transform, 8);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

// Benchmark TBB implementation with different grain sizes
BENCHMARK_DEFINE_F(TransformCloud, CustomTransformTBB_GS100)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudTBB(*cloud, *transformed_cloud, transform, 100);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

BENCHMARK_DEFINE_F(TransformCloud, CustomTransformTBB_GS1000)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudTBB(*cloud, *transformed_cloud, transform, 1000);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

BENCHMARK_DEFINE_F(TransformCloud, CustomTransformTBB_GS10000)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    transformPointCloudTBB(*cloud, *transformed_cloud, transform, 10000);
    benchmark::DoNotOptimize(*transformed_cloud);
  }
}

// Register benchmarks
BENCHMARK_REGISTER_F(TransformCloud, CustomTransform)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, PCLTransform)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, SequentialTransform)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformOMP_1Thread)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformOMP_2Thread)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformOMP_4Thread)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformOMP_8Thread)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformTBB_GS100)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformTBB_GS1000)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK_REGISTER_F(TransformCloud, CustomTransformTBB_GS10000)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK_MAIN();
