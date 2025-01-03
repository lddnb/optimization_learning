/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-24 11:35:43
 * @ Modified by: lddnb
 * @ Modified time: 2025-01-03 18:13:17
 * @ Description: Test for downsampling functions
 */

#include <execution>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include "optimization_learning/downsampling.hpp"

class DownsamplingTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path + "target.pcd", *cloud);
    LOG(INFO) << "Original cloud size: " << cloud->size();
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  const double leaf_size = 0.1;
};

TEST_F(DownsamplingTest, VoxelGridOMP) {
  // Test OMP voxel grid downsampling
  auto start = std::chrono::high_resolution_clock::now();
  auto downsampled_omp = voxelgrid_sampling_omp<pcl::PointXYZI>(cloud, leaf_size, 4);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Downsampled cloud size (OMP): " << downsampled_omp->size();
  LOG(INFO) << "OMP implementation took " << duration << " us";

  // Compare with PCL's voxel grid filter
  start = std::chrono::high_resolution_clock::now();
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_pcl(new pcl::PointCloud<pcl::PointXYZI>);
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_filter.filter(*downsampled_pcl);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Downsampled cloud size (PCL): " << downsampled_pcl->size();
  LOG(INFO) << "PCL implementation took " << duration << " us";

  // Check if sizes are similar (they might not be exactly the same due to different implementations)
  EXPECT_NEAR(downsampled_omp->size(), downsampled_pcl->size(), downsampled_pcl->size() * 0.1);  // Allow 10% difference

  start = std::chrono::high_resolution_clock::now();
  auto downsampled_pstl = voxelgrid_sampling_pstl<pcl::PointXYZI>(cloud, leaf_size);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Downsampled cloud size (PSTL): " << downsampled_pstl->size();
  LOG(INFO) << "PSTL implementation took " << duration << " us";

  std::vector<Eigen::Vector3d> cloud_eigen(cloud->size());
  std::transform(
    std::execution::par,
    cloud->begin(),
    cloud->end(),
    cloud_eigen.begin(),
    [](const pcl::PointXYZI& point) { return Eigen::Vector3d(point.x, point.y, point.z); });

  auto cloud_small_gicp = std::make_shared<small_gicp::PointCloud>(cloud_eigen);
  start = std::chrono::high_resolution_clock::now();
  auto downsampled_cloud_small_gicp = small_gicp::voxelgrid_sampling_omp(*cloud_small_gicp, leaf_size, 4);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Downsampled cloud size (small_gicp omp): " << small_gicp::traits::size(*downsampled_cloud_small_gicp);
  LOG(INFO) << "small_gicp omp implementation took " << duration << " us";
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_small_gicp_pcl(new pcl::PointCloud<pcl::PointXYZI>);
  for (size_t i = 0; i < small_gicp::traits::size(*downsampled_cloud_small_gicp); i++) {
    auto pt = small_gicp::traits::point(*downsampled_cloud_small_gicp, i);
    cloud_small_gicp_pcl->points.push_back(pcl::PointXYZI(pt[0], pt[1], pt[2], 0));
  }
  LOG(INFO) << "Downsampled cloud size (small_gicp omp): " << cloud_small_gicp_pcl->size();

  start = std::chrono::high_resolution_clock::now();
  auto downsampled_cloud_small_gicp_single = small_gicp::voxelgrid_sampling(*cloud_small_gicp, leaf_size);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Downsampled cloud size (small_gicp single): " << small_gicp::traits::size(*downsampled_cloud_small_gicp_single);
  LOG(INFO) << "small_gicp single implementation took " << duration << " us";

  // pcl::io::savePCDFileBinary("/home/ubuntu/ros_ws/src/optimization_learning/data/downsampled_pstl.pcd", *downsampled_pstl);
  // pcl::io::savePCDFileBinary("/home/ubuntu/ros_ws/src/optimization_learning/data/downsampled_pcl.pcd", *downsampled_pcl);
  
  // // 在PCL中显示点云
  // pcl::visualization::PCLVisualizer viewer("Downsampling Test");
  // viewer.addPointCloud<pcl::PointXYZI>(downsampled_omp, "downsampled_omp");
  // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "downsampled_omp");
  // viewer.addPointCloud<pcl::PointXYZI>(downsampled_pcl, "downsampled_pcl");
  // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "downsampled_pcl");
  // viewer.addPointCloud<pcl::PointXYZI>(cloud_small_gicp_pcl, "downsampled_small_gicp");
  // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "downsampled_small_gicp");
  // viewer.spin();
}

TEST_F(DownsamplingTest, VoxelGridOMP_XYZI) {
  // 测试带强度的点云
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>);
  // ... 测试代码 ...
}

TEST_F(DownsamplingTest, VoxelGridOMP_XYZRGB) {
  // 测试带颜色的点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  // ... 测试代码 ...
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  return RUN_ALL_TESTS();
}
