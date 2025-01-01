#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include "optimization_learning/gicp.hpp"

class CovarianceBenchmark : public benchmark::Fixture
{
public:
  void SetUp(const benchmark::State& state)
  {
    std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
    points.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path + "source.pcd", *points);

    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(points);
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
    voxel_filter.filter(*points);
    LOG_FIRST_N(INFO, 1) << "point size: " << points->size();
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr points;
  RegistrationConfig config;
};

BENCHMARK_DEFINE_F(CovarianceBenchmark, ComputeCovarianceSEQ)(benchmark::State& st)
{
  for (auto _ : st) {
    auto covariances = ComputeCovarianceSEQ<pcl::PointXYZI>(points, config.num_neighbors);
    benchmark::DoNotOptimize(covariances);
  }
}

BENCHMARK_DEFINE_F(CovarianceBenchmark, ComputeCovariancePSTL)(benchmark::State& st)
{
  for (auto _ : st) {
    auto covariances = ComputeCovariancePSTL<pcl::PointXYZI>(points, config.num_neighbors);
    benchmark::DoNotOptimize(covariances);
  }
}

BENCHMARK_DEFINE_F(CovarianceBenchmark, ComputeCovarianceOMP)(benchmark::State& st)
{
  for (auto _ : st) {
    auto covariances = ComputeCovarianceOMP<pcl::PointXYZI>(points, config.num_neighbors);
    benchmark::DoNotOptimize(covariances);
  }
}

BENCHMARK_REGISTER_F(CovarianceBenchmark, ComputeCovarianceSEQ)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_REGISTER_F(CovarianceBenchmark, ComputeCovariancePSTL)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_REGISTER_F(CovarianceBenchmark, ComputeCovarianceOMP)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();