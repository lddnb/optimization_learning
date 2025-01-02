from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('optimization_learning')
    
    # 加载yaml配置文件
    config_file = os.path.join(pkg_dir, 'config', 'odometry.yaml')
    
    # 声明参数
    bag_path = LaunchConfiguration('bag_path')
    declare_bag_path = DeclareLaunchArgument(
        'bag_path',
        default_value='/home/ubuntu/ros_ws/src/optimization_learning/bag/rosbag2_2022_04_14-trucks',
        description='Path to the bag file'
    )

    # 启动rosbag play，使用clock参数
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--clock', '-d 3'],
        output='screen'
    )

    # 启动节点
    odometry_node = Node(
        package='optimization_learning',
        executable='odometry_node',
        name='lidar_odometry',
        parameters=[config_file],  # 使用yaml文件中的参数
        output='screen'
    )

    # 启动rviz2
    rviz_config = os.path.join(pkg_dir, 'rviz', 'odometry.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[
            {'use_sim_time': True},
            {'reliability': 'best_effort'}
        ]
    )

    return LaunchDescription([
        odometry_node,
        rviz_node,
        declare_bag_path,
        bag_play
    ])
