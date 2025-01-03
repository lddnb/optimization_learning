#pragma once

#include <execution>
#include <pcl/common/transforms.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

template <typename PointT>
void transformPointCloudSEQ(
  const pcl::PointCloud<PointT>& cloud,
  pcl::PointCloud<PointT>& transformed_cloud,
  const Eigen::Isometry3f& transform)
{
  if (&cloud != &transformed_cloud) {
    transformed_cloud.header = cloud.header;
    transformed_cloud.points.resize(cloud.points.size());
  }
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    transformed_cloud.points[i] = cloud.points[i];
    transformed_cloud.points[i].getVector3fMap() = transform * cloud.points[i].getVector3fMap();
  }
}

template <typename PointT>
void transformPointCloudPSTL(
  const pcl::PointCloud<PointT>& cloud,
  pcl::PointCloud<PointT>& transformed_cloud,
  const Eigen::Isometry3f& transform)
{
  if (&cloud != &transformed_cloud) {
    transformed_cloud.header = cloud.header;
    transformed_cloud.points.resize(cloud.points.size());
  }
  std::vector<size_t> indices(cloud.points.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(
    std::execution::par,
    indices.begin(),
    indices.end(),
    [&cloud, &transform, &transformed_cloud](size_t i) {
      transformed_cloud.points[i] = cloud.points[i];
      transformed_cloud.points[i].getVector3fMap() = transform * cloud.points[i].getVector3fMap();
    });
}

template <typename PointT>
void transformPointCloudOMP(
  const pcl::PointCloud<PointT>& cloud,
  pcl::PointCloud<PointT>& transformed_cloud,
  const Eigen::Isometry3f& transform,
  int num_threads = 4)
{
  if (&cloud != &transformed_cloud) {
    transformed_cloud.header = cloud.header;
    transformed_cloud.points.resize(cloud.points.size());
  }
#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
  for (int64_t i = 0; i < cloud.points.size(); ++i) {
    transformed_cloud.points[i] = cloud.points[i];
    transformed_cloud.points[i].getVector3fMap() = 
      transform * cloud.points[i].getVector3fMap();
  }
}

template <typename PointT>
void transformPointCloudTBB(
  const pcl::PointCloud<PointT>& cloud,
  pcl::PointCloud<PointT>& transformed_cloud,
  const Eigen::Isometry3f& transform,
  int grain_size = 1000)
{
  if (&cloud != &transformed_cloud) {
    transformed_cloud.header = cloud.header;
    transformed_cloud.points.resize(cloud.points.size());
  }
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, cloud.points.size(), grain_size),
    [&](const tbb::blocked_range<size_t>& range) {
      for (size_t i = range.begin(); i < range.end(); ++i) {
        transformed_cloud.points[i] = cloud.points[i];
        transformed_cloud.points[i].getVector3fMap() = 
          transform * cloud.points[i].getVector3fMap();
      }
    });
}
