#!env python
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import pykitti
import numpy as np
from datetime import datetime
from rosbags.rosbag2 import Writer
from rosbags.serde import serialize_cdr
from rosbags.typesys import get_types_from_idl, register_types, get_typestore, Stores
from rosbags.typesys.stores.ros2_jazzy import (
    builtin_interfaces__msg__Time as Time,
    sensor_msgs__msg__Imu as Imu,
    sensor_msgs__msg__PointCloud2 as PointCloud2,
    sensor_msgs__msg__CameraInfo as CameraInfo,
    sensor_msgs__msg__Image as Image,
    geometry_msgs__msg__TransformStamped as TransformStamped,
    nav_msgs__msg__Path as Path,
    sensor_msgs__msg__NavSatFix as NavSatFix,
    geometry_msgs__msg__TwistStamped as TwistStamped,
    std_msgs__msg__Header as Header,
    geometry_msgs__msg__Vector3 as Vector3,
    geometry_msgs__msg__Quaternion as Quaternion,
    sensor_msgs__msg__PointField as PointField,
)
from tqdm import tqdm
import argparse
import math  # 添加这行

typestore = get_typestore(Stores.ROS2_HUMBLE)

# 添加必要的工具函数
def get_time_msg(timestamp):
    """Convert datetime to ROS2 Time message"""
    if isinstance(timestamp, datetime):
        time_sec = float(timestamp.strftime("%s.%f"))
    else:
        time_sec = timestamp
    sec = int(time_sec)
    nanosec = int((time_sec - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert euler angles to quaternion.
    
    Args:
        roll (float): rotation around x-axis (radians)
        pitch (float): rotation around y-axis (radians)
        yaw (float): rotation around z-axis (radians)
        
    Returns:
        list: [x, y, z, w] quaternion
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return [x, y, z, w]

def save_imu_data_raw(writer, kitti, connection, imu_frame_id, topic):
    print("Exporting IMU Raw")
    synced_path = kitti.data_path
    unsynced_path = synced_path.replace('sync', 'extract')
    imu_path = os.path.join(unsynced_path, 'oxts')

    # 读取时间戳
    with open(os.path.join(imu_path, 'timestamps.txt')) as f:
        lines = f.readlines()
        imu_datetimes = []
        for line in lines:
            if len(line) == 1:
                continue
            timestamp = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            imu_datetimes.append(float(timestamp.strftime("%s.%f")))

    # 使用线性模型修正IMU时间
    imu_index = np.asarray(range(len(imu_datetimes)), dtype=np.float64)
    z = np.polyfit(imu_index, imu_datetimes, 1)
    imu_datetimes_new = z[0] * imu_index + z[1]
    imu_datetimes = imu_datetimes_new.tolist()

    # 获取所有IMU数据
    imu_data_dir = os.path.join(imu_path, 'data')
    imu_filenames = sorted(os.listdir(imu_data_dir))
    imu_data = [None] * len(imu_filenames)
    for i, imu_file in enumerate(imu_filenames):
        imu_data_file = open(os.path.join(imu_data_dir, imu_file), "r")
        for line in imu_data_file:
            if len(line) == 1:
                continue
            stripped_line = line.strip()
            line_list = stripped_line.split()
            imu_data[i] = line_list

    assert len(imu_datetimes) == len(imu_data)

    for timestamp, data in zip(imu_datetimes, imu_data):
        roll, pitch, yaw = float(data[3]), float(data[4]), float(data[5])
        
        # 使用我们自己实现的函数计算四元数
        q = quaternion_from_euler(roll, pitch, yaw)
        
        # 创建IMU消息
        imu_msg = Imu(
            header=Header(
                frame_id=imu_frame_id,
                stamp=get_time_msg(timestamp)
            ),
            orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            orientation_covariance=np.zeros(9, dtype=np.float64),
            linear_acceleration=Vector3(
                x=float(data[11]),
                y=float(data[12]), 
                z=float(data[13])
            ),
            linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
            angular_velocity=Vector3(
                x=float(data[17]),
                y=float(data[18]),
                z=float(data[19])
            ),
            angular_velocity_covariance=np.zeros(9, dtype=np.float64),
        )
        
        # 写入bag文件，将秒转换为纳秒
        timestamp_ns = int(timestamp * 1e9)  # 转换为纳秒
        writer.write(connection, timestamp_ns, typestore.serialize_cdr(imu_msg, imu_msg.__msgtype__))

def save_velo_data(writer, kitti, connection, velo_frame_id, topic):
    print("Exporting velodyne data")
    velo_path = os.path.join(kitti.data_path, 'velodyne_points')
    velo_data_dir = os.path.join(velo_path, 'data')
    velo_filenames = sorted(os.listdir(velo_data_dir))
    
    with open(os.path.join(velo_path, 'timestamps_start.txt')) as f:
        lines = f.readlines()
        velo_datetimes = []
        for line in lines:
            if len(line) == 1:
                continue
            dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            velo_datetimes.append(dt)

    for dt, filename in tqdm(zip(velo_datetimes, velo_filenames)):
        if dt is None:
            continue

        velo_filename = os.path.join(velo_data_dir, filename)
        scan = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4)

        # 计算ring通道
        depth = np.linalg.norm(scan, 2, axis=1)
        pitch = np.arcsin(scan[:, 2] / depth)
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        proj_y = (pitch + abs(fov_down)) / fov
        proj_y *= 64
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(64 - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)
        
        # 添加 PointField 数据类型常量
        FLOAT32 = 7  # PointField.FLOAT32
        UINT16 = 3   # PointField.UINT16
        
        fields = [
            PointField(name='x', offset=0, datatype=FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=UINT16, count=1)
        ]

        # 准备点云数据
        point_step = 18  # 每个点的字节数：4*4(float32) + 2(uint16)
        row_step = point_step * len(scan)
        
        # 创建uint8数组来存储所有数据
        cloud_data = np.zeros(row_step, dtype=np.uint8)
        
        # 复制x, y, z, intensity数据
        for i in range(len(scan)):
            # 复制4个float32值
            cloud_data[i*point_step:i*point_step+16].view(np.float32)[:] = scan[i]
            # 复制ring值（uint16）
            cloud_data[i*point_step+16:i*point_step+18].view(np.uint16)[:] = proj_y[i]
        
        # 创建 PointCloud2 消息
        pc_msg = PointCloud2(
            header=Header(
                frame_id=velo_frame_id,
                stamp=get_time_msg(dt)
            ),
            height=1,
            width=len(scan),
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=row_step,
            data=cloud_data,
            is_dense=True
        )
        
        # 写入bag文件，将秒转换为纳秒
        timestamp_ns = int(float(dt.strftime("%s.%f")) * 1e9)  # 转换为纳秒
        writer.write(connection, timestamp_ns, typestore.serialize_cdr(pc_msg, pc_msg.__msgtype__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KITTI dataset to ROS2 bag file!")
    # 添加命令行参数
    kitti_types = ["raw_synced", "odom_color", "odom_gray"]
    odometry_sequences = []
    for s in range(22):
        odometry_sequences.append(str(s).zfill(2))
    
    parser.add_argument("kitti_type", choices=kitti_types, help="KITTI dataset type")
    parser.add_argument("dir", nargs="?", default=os.getcwd(), help="base directory of the dataset")
    parser.add_argument("-t", "--date", help="date of the raw dataset (i.e. 2011_09_26)")
    parser.add_argument("-r", "--drive", help="drive number of the raw dataset (i.e. 0001)")
    parser.add_argument("-s", "--sequence", choices=odometry_sequences, help="sequence of the odometry dataset (00-21)")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), "ros2_bags")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建bag文件名
    if args.kitti_type.find("raw") != -1:
        bag_name = f"kitti_{args.date}_drive_{args.drive}_{args.kitti_type[4:]}"
    else:
        bag_name = f"kitti_data_odometry_{args.kitti_type[5:]}_sequence_{args.sequence}"
    
    bag_path = os.path.join(output_dir, bag_name)
    
    # 创建Writer对象，使用正确的配置
    with Writer(bag_path, version=9) as writer:
        # 创建连接
        connections = [
            writer.add_connection('/kitti/oxts/imu', Imu.__msgtype__, typestore=typestore),
            writer.add_connection('/kitti/velo', PointCloud2.__msgtype__, typestore=typestore),
            writer.add_connection('/kitti/oxts/gps/fix', NavSatFix.__msgtype__, typestore=typestore),
            writer.add_connection('/kitti/oxts/gps/vel', TwistStamped.__msgtype__, typestore=typestore),
            writer.add_connection('/kitti/ground_truth', Path.__msgtype__, typestore=typestore),
        ]

        # 处理数据并写入bag
        if args.kitti_type.find("raw") != -1:
            # 处理raw数据集
            kitti = pykitti.raw(args.dir, args.date, args.drive)
            
            if not os.path.exists(kitti.data_path):
                print('Path {} does not exists. Exiting.'.format(kitti.data_path))
                sys.exit(1)

            if len(kitti.timestamps) == 0:
                print('Dataset is empty? Exiting.')
                sys.exit(1)
                
            # 写入各类数据
            save_imu_data_raw(writer, kitti, connections[0], 'imu_link', '/kitti/oxts/imu')
            save_velo_data(writer, kitti, connections[1], 'velo_link', '/kitti/velo')
