/**
 * @file cloud_process.hpp
 * @author lddnb (lz750126471@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "optimization_learning/common.hpp"

template<typename PointT>
void VelodyneHandler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg, pcl::PointCloud<PointT>& cloud_out) {
    int num_scans_ = 32;
    int point_filter_num_ = 1;
    cloud_out.clear();

    pcl::PointCloud<VelodynePoint> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out.reserve(plsize);

    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time

    bool given_offset_time = false;
    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time = true;
    } else {
        given_offset_time = false;

        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointT added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.time = pl_orig.points[i].time * 1e-6; // curvature unit: us

        /// 略掉过近的点
        if (added_pt.getVector3fMap().norm() < 2.0) {
            continue;
        }

        if (!given_offset_time) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.time = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.time;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.time = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.time = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.time < time_last[layer]) {
                added_pt.time += 360.0 / omega_l;
            }

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.time;
        }

        if (i % point_filter_num_ == 0) {
            cloud_out.points.emplace_back(added_pt);
        }
    }
}