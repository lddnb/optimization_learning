/**
 * @file downsampling.hpp
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
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

inline Eigen::Array3i fast_floor(const Eigen::Array3f& pt)
{
  const Eigen::Array3i ncoord = pt.cast<int>();
  return ncoord - (pt < ncoord.cast<float>()).cast<int>();
}

/// @brief Implementation of quick sort with OpenMP parallelism. Do not call this directly. Use quick_sort_omp instead.
/// @param first  First iterator
/// @param last   Last iterator
/// @param comp   Comparison function
template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp_impl(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp) {
  const std::ptrdiff_t n = std::distance(first, last);
  if (n < 1024) {
    std::sort(first, last, comp);
    return;
  }

  const auto median3 = [&](const auto& a, const auto& b, const auto& c, const Compare& comp) {
    return comp(a, b) ? (comp(b, c) ? b : (comp(a, c) ? c : a)) : (comp(a, c) ? a : (comp(b, c) ? c : b));
  };

  const int offset = n / 8;
  const auto m1 = median3(*first, *(first + offset), *(first + offset * 2), comp);
  const auto m2 = median3(*(first + offset * 3), *(first + offset * 4), *(first + offset * 5), comp);
  const auto m3 = median3(*(first + offset * 6), *(first + offset * 7), *(last - 1), comp);

  auto pivot = median3(m1, m2, m3, comp);
  auto middle1 = std::partition(first, last, [&](const auto& val) { return comp(val, pivot); });
  auto middle2 = std::partition(middle1, last, [&](const auto& val) { return !comp(pivot, val); });

#pragma omp task
  quick_sort_omp_impl(first, middle1, comp);

#pragma omp task
  quick_sort_omp_impl(middle2, last, comp);
}

/// @brief Quick sort with OpenMP parallelism.
/// @param first        First iterator
/// @param last         Last iterator
/// @param comp         Comparison function
/// @param num_threads  Number of threads
template <typename RandomAccessIterator, typename Compare>
void quick_sort_omp(RandomAccessIterator first, RandomAccessIterator last, const Compare& comp, int num_threads) {
#ifndef _MSC_VER
#pragma omp parallel num_threads(num_threads)
  {
#pragma omp single nowait
    { quick_sort_omp_impl(first, last, comp); }
  }
#else
  std::sort(first, last, comp);
#endif
}

// 添加点云加法操作的辅助函数
template<typename PointT>
void accumulate_point(PointT& sum_pt, const PointT& pt) {
  sum_pt.getVector3fMap() += pt.getVector3fMap();
  
  if constexpr (pcl::traits::has_field<PointT, pcl::fields::intensity>::value) {
    sum_pt.intensity += pt.intensity;
  }
  if constexpr (pcl::traits::has_field<PointT, pcl::fields::rgb>::value) {
    float r = static_cast<float>((sum_pt.rgb >> 16) & 0xFF) + static_cast<float>((pt.rgb >> 16) & 0xFF);
    float g = static_cast<float>((sum_pt.rgb >> 8) & 0xFF) + static_cast<float>((pt.rgb >> 8) & 0xFF);
    float b = static_cast<float>(sum_pt.rgb & 0xFF) + static_cast<float>(pt.rgb & 0xFF);
    sum_pt.rgb = (static_cast<uint32_t>(r) << 16) | 
                 (static_cast<uint32_t>(g) << 8) | 
                 static_cast<uint32_t>(b);
  }
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
voxelgrid_sampling_omp(const typename pcl::PointCloud<PointT>::Ptr& points, double leaf_size, int num_threads = 4)
{
  if (points->empty()) {
    return std::make_shared<pcl::PointCloud<PointT>>();
  }

  const double inv_leaf_size = 1.0 / leaf_size;

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3 = 63bits in 64bit int)
  constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive

  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(points->size());
#pragma omp parallel for num_threads(num_threads) schedule(guided, 32)
  for (std::int64_t i = 0; i < points->size(); i++) {
    const Eigen::Array3f pt_f = points->points[i].getVector3fMap();
    const Eigen::Array3i coord = fast_floor(pt_f * inv_leaf_size) + coord_offset;
    if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
      std::cerr << "warning: voxel coord is out of range!!" << std::endl;
      coord_pt[i] = {invalid_coord, i};
      continue;
    }
    // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
    const std::uint64_t bits =                                                           //
      (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |  //
      (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |  //
      (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
    coord_pt[i] = {bits, i};
  }

  // Sort by voxel coords
  quick_sort_omp(coord_pt.begin(), coord_pt.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }, num_threads);

  auto downsampled = std::make_shared<pcl::PointCloud<PointT>>();
  downsampled->resize(points->size());

  // Take block-wise sum
  const int block_size = 1024;
  std::atomic_uint64_t num_points = 0;

#pragma omp parallel for num_threads(num_threads) schedule(guided, 4)
  for (std::int64_t block_begin = 0; block_begin < points->size(); block_begin += block_size) {
    std::vector<PointT> sub_points;
    sub_points.reserve(block_size);

    const size_t block_end = std::min<size_t>(points->size(), block_begin + block_size);

    PointT sum_pt = points->points[coord_pt[block_begin].second];
    int sum_num = 1;

    for (size_t i = block_begin + 1; i != block_end; i++) {
      if (coord_pt[i].first == invalid_coord) {
        continue;
      }

      if (coord_pt[i - 1].first != coord_pt[i].first) {
        // 计算平均值
        float inv_sum_num = 1.0f / sum_num;
        PointT avg_pt;
        // xyz坐标取平均
        avg_pt.getVector3fMap() = sum_pt.getVector3fMap() * inv_sum_num;
        // 其他字段也取平均
        if constexpr (pcl::traits::has_field<PointT, pcl::fields::intensity>::value) {
          avg_pt.intensity = sum_pt.intensity * inv_sum_num;
        }
        if constexpr (pcl::traits::has_field<PointT, pcl::fields::rgb>::value) {
          float r = static_cast<float>((sum_pt.rgb >> 16) & 0xFF) * inv_sum_num;
          float g = static_cast<float>((sum_pt.rgb >> 8) & 0xFF) * inv_sum_num;
          float b = static_cast<float>(sum_pt.rgb & 0xFF) * inv_sum_num;
          avg_pt.rgb = (static_cast<uint32_t>(r) << 16) | 
                      (static_cast<uint32_t>(g) << 8) | 
                      static_cast<uint32_t>(b);
        }
        sub_points.emplace_back(avg_pt);
        
        // 重置累加器
        sum_pt = points->points[coord_pt[i].second];
        sum_num = 1;
      } else {
        // 累加点
        accumulate_point(sum_pt, points->points[coord_pt[i].second]);
        sum_num++;
      }
    }

    // 处理最后一个体素
    if (sum_num > 0) {
      float inv_sum_num = 1.0f / sum_num;
      PointT avg_pt;
      avg_pt.getVector3fMap() = sum_pt.getVector3fMap() * inv_sum_num;
      if constexpr (pcl::traits::has_field<PointT, pcl::fields::intensity>::value) {
        avg_pt.intensity = sum_pt.intensity * inv_sum_num;
      }
      if constexpr (pcl::traits::has_field<PointT, pcl::fields::rgb>::value) {
        float r = static_cast<float>((sum_pt.rgb >> 16) & 0xFF) * inv_sum_num;
        float g = static_cast<float>((sum_pt.rgb >> 8) & 0xFF) * inv_sum_num;
        float b = static_cast<float>(sum_pt.rgb & 0xFF) * inv_sum_num;
        avg_pt.rgb = (static_cast<uint32_t>(r) << 16) | 
                    (static_cast<uint32_t>(g) << 8) | 
                    static_cast<uint32_t>(b);
      }
      sub_points.emplace_back(avg_pt);
    }

    const size_t point_index_begin = num_points.fetch_add(sub_points.size());
    for (size_t i = 0; i < sub_points.size(); i++) {
      downsampled->points[point_index_begin + i] = sub_points[i];
    }
  }

  downsampled->resize(num_points);
  return downsampled;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
voxelgrid_sampling_pstl(const typename pcl::PointCloud<PointT>::Ptr& points, double leaf_size)
{
  if (points->empty()) {
    return std::make_shared<pcl::PointCloud<PointT>>();
  }

  const double inv_leaf_size = 1.0 / leaf_size;

  constexpr std::uint64_t invalid_coord = std::numeric_limits<std::uint64_t>::max();
  constexpr int coord_bit_size = 21;                       // Bits to represent each voxel coordinate (pack 21x3 = 63bits in 64bit int)
  constexpr size_t coord_bit_mask = (1 << 21) - 1;         // Bit mask
  constexpr int coord_offset = 1 << (coord_bit_size - 1);  // Coordinate offset to make values positive

  // 使用vector存储体素坐标和点索引
  std::vector<std::pair<std::uint64_t, size_t>> coord_pt(points->size());
  
  // 并行计算体素坐标
  std::for_each(
    std::execution::par,
    std::begin(coord_pt),
    std::end(coord_pt),
    [&](auto& pair) {
      const size_t i = &pair - coord_pt.data();
      const Eigen::Array3f pt_f = points->points[i].getVector3fMap();
      const Eigen::Array3i coord = fast_floor(pt_f * inv_leaf_size) + coord_offset;
      if ((coord < 0).any() || (coord > coord_bit_mask).any()) {
        std::cerr << "warning: voxel coord is out of range!!" << std::endl;
        pair = {invalid_coord, i};
        return;
      }
      // Compute voxel coord bits (0|1bit, z|21bit, y|21bit, x|21bit)
      const std::uint64_t bits =                                                           
        (static_cast<std::uint64_t>(coord[0] & coord_bit_mask) << (coord_bit_size * 0)) |  
        (static_cast<std::uint64_t>(coord[1] & coord_bit_mask) << (coord_bit_size * 1)) |  
        (static_cast<std::uint64_t>(coord[2] & coord_bit_mask) << (coord_bit_size * 2));
      pair = {bits, i};
    }
  );

  // 并行排序
  std::sort(
    std::execution::par,
    coord_pt.begin(),
    coord_pt.end(),
    [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }
  );

  // 找到所有不同的体素坐标
  std::vector<size_t> voxel_starts;  // 存储每个体素的起始索引
  voxel_starts.push_back(0);
  for (size_t i = 1; i < coord_pt.size(); ++i) {
    if (coord_pt[i].first != coord_pt[i-1].first) {
      voxel_starts.push_back(i);
    }
  }
  voxel_starts.push_back(coord_pt.size());  // 添加结束索引

  // 为每个体素创建一个结果，使用正确的分配器
  std::vector<PointT, Eigen::aligned_allocator<PointT>> temp_points(voxel_starts.size() - 1);

  // 并行处理每个体素
  std::for_each(
    std::execution::par,
    std::begin(voxel_starts),
    std::prev(std::end(voxel_starts)),
    [&](const auto& start_idx) {
      const size_t voxel_idx = &start_idx - voxel_starts.data();
      const size_t end_idx = voxel_starts[voxel_idx + 1];
      
      if (coord_pt[start_idx].first == invalid_coord) {
        return;
      }

      // 计算这个体素内所有点的平均值
      auto sum_pt = PointT();
      // 顺序处理同一体素内的点
      for (size_t i = start_idx; i < end_idx; ++i) {
        accumulate_point(sum_pt, points->points[coord_pt[i].second]);
      }

      // 计算平均值
      int sum_num = end_idx - start_idx;
      float inv_sum_num = 1.0f / sum_num;
      PointT& avg_pt = temp_points[voxel_idx];
      avg_pt.getVector3fMap() = sum_pt.getVector3fMap() * inv_sum_num;
      if constexpr (pcl::traits::has_field<PointT, pcl::fields::intensity>::value) {
        avg_pt.intensity = sum_pt.intensity * inv_sum_num;
      }
      if constexpr (pcl::traits::has_field<PointT, pcl::fields::rgb>::value) {
        float r = static_cast<float>((sum_pt.rgb >> 16) & 0xFF) * inv_sum_num;
        float g = static_cast<float>((sum_pt.rgb >> 8) & 0xFF) * inv_sum_num;
        float b = static_cast<float>(sum_pt.rgb & 0xFF) * inv_sum_num;
        avg_pt.rgb = (static_cast<uint32_t>(r) << 16) | 
                    (static_cast<uint32_t>(g) << 8) | 
                    static_cast<uint32_t>(b);
      }
    }
  );

  // 移除无效的点
  temp_points.erase(
    std::remove_if(
      temp_points.begin(),
      temp_points.end(),
      [](const PointT& pt) {
        // 检查是否是未被处理的默认点
        return pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f;
      }),
    temp_points.end());

  // 创建输出点云
  auto downsampled = std::make_shared<pcl::PointCloud<PointT>>();
  downsampled->points = std::move(temp_points);
  downsampled->width = downsampled->points.size();
  downsampled->height = 1;
  downsampled->is_dense = true;

  return downsampled;
}