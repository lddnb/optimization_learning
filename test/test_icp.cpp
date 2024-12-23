/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-24 11:35:43
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-24 12:21:50
 * @ Description:
 */

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"

class ICPTest : public ::testing::Test {
protected:
  void SetUp() override {
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

    LOG(INFO) << "source points size: " << source_points->size();
    LOG(INFO) << "target points size: " << target_points->size();
    
    R_true = Eigen::Quaterniond(Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitX()));
    t_true = Eigen::Vector3d(1, 2, 3);
    T_true = Eigen::Affine3d(Eigen::Translation3d(t_true) * R_true.toRotationMatrix());
    pcl::transformPointCloud(*target_points, *target_points, T_true);

    R_init = Eigen::Quaterniond(Eigen::AngleAxisd(1.45, Eigen::Vector3d::UnitX()));
    t_init = Eigen::Vector3d(1.2, 2.2, 3.2);
    T_init = Eigen::Affine3d(Eigen::Translation3d(t_init) * R_init.toRotationMatrix());

    LOG(INFO) << "R_init: " << R_init.coeffs().transpose();
    LOG(INFO) << "t_init: " << t_init.transpose();
    LOG(INFO) << "R_Ture: " << R_true.coeffs().transpose();
    LOG(INFO) << "t_ture: " << t_true.transpose();
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
};

TEST_F(ICPTest, PointToPointICP) {
  LOG(INFO) << "======================== Point to Point ICP ========================";
  double R_err = 0.1;
  double t_err = 0.3;

  Eigen::Affine3d T_opt;
  int iterations;

  LOG(INFO) << "------------------- Ceres ------------------";
  auto start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PICP_Ceres<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- GTSAM SE3 ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PICP_GTSAM_SE3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- GTSAM SO3+R3 ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PICP_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- GN ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PICP_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- PCL ICP ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PICP_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err); 
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- small_gicp icp ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  config.rotation_eps = 0.1 * M_PI / 180.0;
  ICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);
}

TEST_F(ICPTest, PointToPlaneICP) {
  LOG(INFO) << "======================== Point to Plane ICP ========================";
  double R_err = 0.1;
  double t_err = 0.1;
  
  Eigen::Affine3d T_opt;
  int iterations;

  LOG(INFO) << "------------------- Ceres ------------------";
  auto start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_Ceres<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- GTSAM SE3 ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_GTSAM_SE3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- GTSAM SO3+R3 ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- GN ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- PCL NICP ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);

  LOG(INFO) << "------------------- small_gicp point to plane icp ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  config2.rotation_eps = 0.1 * M_PI / 180.0;
  P2PlaneICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config2);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t: " << T_opt.translation().transpose();
  EXPECT_NEAR((Eigen::Quaterniond(T_opt.rotation()).coeffs() - R_true.coeffs()).norm(), 0, R_err);
  EXPECT_NEAR((T_opt.translation() - t_true).norm(), 0, t_err);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  return RUN_ALL_TESTS();
} 