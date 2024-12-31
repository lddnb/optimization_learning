#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"
#include "optimization_learning/gicp.hpp"
#include "optimization_learning/ndt.hpp"

class ICP : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State& state) {
    std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
    source_points.reset(new pcl::PointCloud<pcl::PointXYZI>);
    target_points.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path + "source.pcd", *source_points);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path + "target.pcd", *target_points);

    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(source_points);
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
    voxel_filter.filter(*source_points);
    voxel_filter.setInputCloud(target_points);
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
    voxel_filter.filter(*target_points);

    R_true = Eigen::Quaterniond(Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitX()));
    t_true = Eigen::Vector3d(1, 2, 3);
    T_true = Eigen::Affine3d(Eigen::Translation3d(t_true) * R_true.toRotationMatrix());
    pcl::transformPointCloud(*target_points, *target_points, T_true);

    R_init = Eigen::Quaterniond(Eigen::AngleAxisd(1.45, Eigen::Vector3d::UnitX()));
    t_init = Eigen::Vector3d(1.2, 2.2, 3.2);
    T_init = Eigen::Affine3d(Eigen::Translation3d(t_init) * R_init.toRotationMatrix());
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_points;
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_points;
  Eigen::Quaterniond R_true;
  Eigen::Vector3d t_true;
  Eigen::Affine3d T_true;
  Eigen::Quaterniond R_init;
  Eigen::Vector3d t_init;
  Eigen::Affine3d T_init;
  ICPConfig config;
  PointToPlaneICPConfig config2;
  GICPConfig config3;
};

// Point to Point ICP Benchmarks
BENCHMARK_DEFINE_F(ICP, P2PICP_Ceres)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PICP_Ceres<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PICP_GTSAM_SE3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PICP_GTSAM_SE3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PICP_GTSAM_SO3_R3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PICP_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PICP_GN)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PICP_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PICP_PCL)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PICP_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, ICP_small_gicp)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    ICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
    benchmark::DoNotOptimize(T_opt);
  }
}

// Point to Plane ICP Benchmarks
BENCHMARK_DEFINE_F(ICP, P2PlaneICP_Ceres)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PlaneICP_Ceres<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PlaneICP_GTSAM_SE3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PlaneICP_GTSAM_SE3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PlaneICP_GTSAM_SO3_R3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PlaneICP_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PlaneICP_GN)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PlaneICP_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PlaneICP_PCL)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PlaneICP_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, P2PlaneICP_small_gicp)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    P2PlaneICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
    benchmark::DoNotOptimize(T_opt);
  }
}

// GICP Benchmarks
BENCHMARK_DEFINE_F(ICP, GICP_Ceres)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    GICP_Ceres<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, GICP_GTSAM_SE3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    GICP_GTSAM_SE3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, GICP_GTSAM_SO3_R3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    GICP_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, GICP_GN)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    GICP_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, GICP_PCL)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    GICP_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(ICP, GICP_small_gicp)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = T_init;
    int iterations;
    GICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

class NDT : public benchmark::Fixture {
public:
  void SetUp(const benchmark::State& state) {
    source_points.reset(new pcl::PointCloud<pcl::PointXYZI>);
    target_points.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(
      "/home/ubuntu/ros_ws/src/optimization_learning/thirdparty/ndt_omp/data/251370668.pcd",
      *source_points);
    pcl::io::loadPCDFile<pcl::PointXYZI>(
      "/home/ubuntu/ros_ws/src/optimization_learning/thirdparty/ndt_omp/data/251371071.pcd",
      *target_points);

    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(source_points);
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
    voxel_filter.filter(*source_points);
    voxel_filter.setInputCloud(target_points);
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);
    voxel_filter.filter(*target_points);
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_points;
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_points;
  NDTConfig config3;
};

// NDT Benchmarks
BENCHMARK_DEFINE_F(NDT, NDT_Ceres)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = Eigen::Affine3d::Identity();
    int iterations;
    NDT_Ceres<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(NDT, NDT_GTSAM_SE3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = Eigen::Affine3d::Identity();
    int iterations;
    NDT_GTSAM_SE3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(NDT, NDT_GTSAM_SO3_R3)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = Eigen::Affine3d::Identity();
    int iterations;
    NDT_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(NDT, NDT_GN)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = Eigen::Affine3d::Identity();
    int iterations;
    NDT_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(NDT, NDT_PCL)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = Eigen::Affine3d::Identity();
    int iterations;
    NDT_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

BENCHMARK_DEFINE_F(NDT, NDT_OMP)(benchmark::State& st) {
  for (auto _ : st) {
    Eigen::Affine3d T_opt = Eigen::Affine3d::Identity();
    int iterations;
    NDT_OMP<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config3);
    benchmark::DoNotOptimize(T_opt);
  }
}

// Register all benchmarks
BENCHMARK_REGISTER_F(ICP, P2PICP_Ceres)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PICP_GTSAM_SE3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PICP_GTSAM_SO3_R3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PICP_GN)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PICP_PCL)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, ICP_small_gicp)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_REGISTER_F(ICP, P2PlaneICP_Ceres)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PlaneICP_GTSAM_SE3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PlaneICP_GTSAM_SO3_R3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PlaneICP_GN)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PlaneICP_PCL)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, P2PlaneICP_small_gicp)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_REGISTER_F(ICP, GICP_Ceres)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, GICP_GTSAM_SE3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, GICP_GTSAM_SO3_R3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, GICP_GN)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, GICP_PCL)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(ICP, GICP_small_gicp)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_REGISTER_F(NDT, NDT_Ceres)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(NDT, NDT_GTSAM_SE3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(NDT, NDT_GTSAM_SO3_R3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(NDT, NDT_GN)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(NDT, NDT_PCL)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_REGISTER_F(NDT, NDT_OMP)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();