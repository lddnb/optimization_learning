#!/usr/bin/python

import sys
import numpy as np
from scipy.spatial.transform import Rotation

def transform_to_origin(poses):
    """将轨迹原点归零，其他点相应变换"""
    # 获取第一个位姿
    first_pose = poses[0]
    first_pos = first_pose[:3]
    first_rpy = first_pose[3:]
    
    # 创建初始位姿的变换矩阵
    first_rot = Rotation.from_euler('xyz', first_rpy).as_matrix()
    T_init = np.eye(4)
    T_init[:3, :3] = first_rot
    T_init[:3, 3] = first_pos
    
    # 计算初始位姿的逆
    T_init_inv = np.linalg.inv(T_init)
    
    # 创建绕y=x轴旋转180度的变换矩阵
    # 先交换x和y，再取反
    flip_matrix = np.array([
        [0, -1,  0, 0],  # x变为-y
        [-1, 0,  0, 0],  # y变为-x
        [0,  0,  1, 0],  # z保持不变
        [0,  0,  0, 1]
    ])
    
    # 变换所有位姿
    transformed_poses = []
    for pose in poses:
        # 创建当前位姿的变换矩阵
        pos = pose[:3]
        rpy = pose[3:]
        rot = Rotation.from_euler('xyz', rpy).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        
        # 应用变换：先归零，再绕y=x轴翻转
        T_new = flip_matrix @ T_init_inv @ T
        
        # 提取新的位置和姿态
        new_pos = T_new[:3, 3]
        new_rot = Rotation.from_matrix(T_new[:3, :3]).as_euler('xyz')
        
        transformed_poses.append(np.concatenate([new_pos, new_rot]))
    
    return np.array(transformed_poses)

def save_tum_format(timestamps, poses, filename):
    """保存为TUM格式：timestamp tx ty tz qx qy qz qw"""
    with open(filename, 'w') as f:
        for t, pose in zip(timestamps, poses):
            # 转换欧拉角为四元数
            quat = Rotation.from_euler('xyz', pose[3:]).as_quat()
            # 写入TUM格式：timestamp tx ty tz qx qy qz qw
            f.write(f"{t:.6f} {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} "
                   f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")

def main(args):
    if len(sys.argv) < 2:
        print('Usage: python read_ground_truth.py groundtruth.csv')
        return 1

    # 读取数据
    gt = np.loadtxt(sys.argv[1], delimiter=",")

    timestamps = gt[:, 0]  # 提取时间戳
    poses = gt[:, 1:]      # 提取位姿数据 [x,y,z,roll,pitch,yaw]

    # 将轨迹原点归零
    transformed_poses = transform_to_origin(poses)

    # 保存为TUM格式
    save_tum_format(timestamps, transformed_poses, "transformed_trajectory.txt")

    print(f"Transformed trajectory saved to 'transformed_trajectory.txt'")

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
