/**
 * @file downsampling_benchmark.cpp
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
#include <pcl/filters/voxel_grid.h>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include "optimization_learning/downsampling.hpp"

class Downsampling : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State& state) {
    std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path + "target.pcd", *cloud);

    LOG_FIRST_N(INFO, 1) << "pcl cloud size: " << cloud->size();

    std::vector<Eigen::Vector3d> cloud_eigen(cloud->size());
    std::transform(
      std::execution::par,
      cloud->begin(),
      cloud->end(),
      cloud_eigen.begin(),
      [](const pcl::PointXYZ& point) { return Eigen::Vector3d(point.x, point.y, point.z); });
    cloud_small_gicp = std::make_shared<small_gicp::PointCloud>(cloud_eigen);

    LOG_FIRST_N(INFO, 1) << "small_gicp cloud size: " << small_gicp::traits::size(*cloud_small_gicp);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  const double leaf_size = 0.1;
  std::shared_ptr<small_gicp::PointCloud> cloud_small_gicp;
};

// Benchmark OMP voxel grid with different thread counts
BENCHMARK_DEFINE_F(Downsampling, VoxelGridOMP_1Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = voxelgrid_sampling_omp<pcl::PointXYZ>(cloud, leaf_size, 1);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridOMP_2Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = voxelgrid_sampling_omp<pcl::PointXYZ>(cloud, leaf_size, 2);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridOMP_4Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = voxelgrid_sampling_omp<pcl::PointXYZ>(cloud, leaf_size, 4);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridOMP_8Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = voxelgrid_sampling_omp<pcl::PointXYZ>(cloud, leaf_size, 8);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridSmallGICP)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = small_gicp::voxelgrid_sampling(*cloud_small_gicp, leaf_size);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridSmallGICPOMP_1Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = small_gicp::voxelgrid_sampling_omp(*cloud_small_gicp, leaf_size, 1);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridSmallGICPOMP_2Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = small_gicp::voxelgrid_sampling_omp(*cloud_small_gicp, leaf_size, 2);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridSmallGICPOMP_4Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = small_gicp::voxelgrid_sampling_omp(*cloud_small_gicp, leaf_size, 4);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridSmallGICPOMP_8Thread)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = small_gicp::voxelgrid_sampling_omp(*cloud_small_gicp, leaf_size, 8);
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK_DEFINE_F(Downsampling, VoxelGridPSTL)(benchmark::State& st) {
  for (auto _ : st) {
    auto result = voxelgrid_sampling_pstl<pcl::PointXYZ>(cloud, leaf_size);
    benchmark::DoNotOptimize(result);
  }
}

// Benchmark PCL's voxel grid for comparison
BENCHMARK_DEFINE_F(Downsampling, VoxelGridPCL)(benchmark::State& st) {
  for (auto _ : st) {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*result);
    benchmark::DoNotOptimize(result);
  }
}

// Register benchmarks
BENCHMARK_REGISTER_F(Downsampling, VoxelGridOMP_1Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridOMP_2Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridOMP_4Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridOMP_8Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridSmallGICP)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridSmallGICPOMP_1Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridSmallGICPOMP_2Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridSmallGICPOMP_4Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridSmallGICPOMP_8Thread)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridPSTL)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(Downsampling, VoxelGridPCL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
