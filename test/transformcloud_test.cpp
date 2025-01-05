/**
 * @file transformcloud_test.cpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "optimization_learning/transform_point_cloud.hpp"

class TransformCloudTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
    cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path + "target.pcd", *cloud);
    LOG(INFO) << "Original cloud size: " << cloud->size();

    // 创建一个随机变换矩阵
    Eigen::AngleAxisf rotation_vector(M_PI / 4, Eigen::Vector3f(1, 0, 0));  // 绕X轴旋转45度
    transform.linear() = rotation_vector.toRotationMatrix();
    transform.translation() = Eigen::Vector3f(1.0, 2.0, 3.0);  // 平移(1,2,3)

    // 使用PCL的函数生成参考结果
    pcl::transformPointCloud(*cloud, *reference_cloud, transform.matrix());
  }

  // 检查两个点云是否相等
  bool cloudsEqual(const pcl::PointCloud<pcl::PointXYZI>& cloud1, 
                  const pcl::PointCloud<pcl::PointXYZI>& cloud2,
                  float epsilon = 1e-5) {
    if (cloud1.size() != cloud2.size()) return false;
    
    for (size_t i = 0; i < cloud1.size(); ++i) {
      if ((cloud1.points[i].getVector3fMap() - cloud2.points[i].getVector3fMap()).norm() > epsilon) {
        return false;
      }
      if (std::abs(cloud1.points[i].intensity - cloud2.points[i].intensity) > epsilon) {
        return false;
      }
    }
    return true;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZI>};
  pcl::PointCloud<pcl::PointXYZI>::Ptr reference_cloud{new pcl::PointCloud<pcl::PointXYZI>};
  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
};

TEST_F(TransformCloudTest, SEQ_Implementation) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr result(new pcl::PointCloud<pcl::PointXYZI>);
  auto start = std::chrono::high_resolution_clock::now();
  transformPointCloudSEQ(*cloud, *result, transform);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "SEQ implementation took " << duration << " us";
  
  EXPECT_TRUE(cloudsEqual(*result, *reference_cloud)) 
    << "SEQ implementation result differs from PCL reference";
  EXPECT_EQ(result->size(), reference_cloud->size()) 
    << "SEQ implementation output size mismatch";
}

TEST_F(TransformCloudTest, PSTL_Implementation) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr result(new pcl::PointCloud<pcl::PointXYZI>);
  auto start = std::chrono::high_resolution_clock::now();
  transformPointCloudPSTL(*cloud, *result, transform);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "PSTL implementation took " << duration << " us";
  
  EXPECT_TRUE(cloudsEqual(*result, *reference_cloud)) 
    << "PSTL implementation result differs from PCL reference";
  EXPECT_EQ(result->size(), reference_cloud->size()) 
    << "PSTL implementation output size mismatch";
}

TEST_F(TransformCloudTest, OMP_Implementation) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr result(new pcl::PointCloud<pcl::PointXYZI>);
  auto start = std::chrono::high_resolution_clock::now();
  transformPointCloudOMP(*cloud, *result, transform, 4);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "OMP implementation took " << duration << " us";
  
  EXPECT_TRUE(cloudsEqual(*result, *reference_cloud)) 
    << "OMP implementation result differs from PCL reference";
  EXPECT_EQ(result->size(), reference_cloud->size()) 
    << "OMP implementation output size mismatch";
}

TEST_F(TransformCloudTest, TBB_Implementation) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr result(new pcl::PointCloud<pcl::PointXYZI>);
  auto start = std::chrono::high_resolution_clock::now();
  transformPointCloudTBB(*cloud, *result, transform, 1000);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "TBB implementation took " << duration << " us";
  
  EXPECT_TRUE(cloudsEqual(*result, *reference_cloud)) 
    << "TBB implementation result differs from PCL reference";
  EXPECT_EQ(result->size(), reference_cloud->size()) 
    << "TBB implementation output size mismatch";
}

// 测试不同线程数的OMP实现
TEST_F(TransformCloudTest, OMP_DifferentThreads) {
  for (int threads : {1, 2, 4, 8}) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr result(new pcl::PointCloud<pcl::PointXYZI>);
    auto start = std::chrono::high_resolution_clock::now();
    transformPointCloudOMP(*cloud, *result, transform, threads);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG(INFO) << "OMP implementation with " << threads << " threads took " << duration << " us";
    EXPECT_TRUE(cloudsEqual(*result, *reference_cloud)) 
      << "OMP implementation with " << threads << " threads differs from PCL reference";
  }
}

// 测试不同grain size的TBB实现
TEST_F(TransformCloudTest, TBB_DifferentGrainSizes) {
  for (int grain_size : {100, 1000, 10000}) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr result(new pcl::PointCloud<pcl::PointXYZI>);
    auto start = std::chrono::high_resolution_clock::now();
    transformPointCloudTBB(*cloud, *result, transform, grain_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOG(INFO) << "TBB implementation with grain size " << grain_size << " took " << duration << " us";
    EXPECT_TRUE(cloudsEqual(*result, *reference_cloud)) 
      << "TBB implementation with grain size " << grain_size << " differs from PCL reference";
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  return RUN_ALL_TESTS();
}
