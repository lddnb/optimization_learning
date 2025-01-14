# optimization_learning

## Dependencies
- ROS2 jazzy
- Eigen 3.4.0
- PCL 1.14.0
- Ceres 2.2.0
- GTSAM 4.2.0
- OpenCV 4.6.0
- evo
  ```bash
  pip install evo --break-system-packages
  ```

## Comparison
### kitti_2011_10_03_0027
#### 1. downsample_timing_comparison
![downsample_timing_comparison](doc/downsample_timing_comparison.png)
#### 2. gicp_traj_comparison
![gicp_traj_comparison](doc/gicp_traj_comparison.png)
#### 3. gicp_timing_comparison
![gicp_timing_comparison_series](doc/gicp_timing_comparison_series.png)
![gicp_timing_comparison_violin](doc/gicp_timing_comparison_violin.png)
#### 4. reg_traj_comparison
![reg_traj_comparison](doc/reg_traj_comparison.png)
#### 5. reg_timing_comparison
![reg_timing_comparison_series](doc/reg_timing_comparison_series.png)
![reg_timing_comparison_violin](doc/reg_timing_comparison_violin.png)

gicp_Ceres failed


## Todo list
- [x]  create dev container
- [x]  Migrate to ROS2 jazzy
- [x]  Add point-to-plane ICP
- [x]  Add NDT 
- [x]  Add GICP
- [ ]  Add ESKF
- [ ]  Add IMU-preintegration
## Question
1. issue #1
2. NICP Epsilon


## References
- [slam_in_autonomous_driving](https://github.com/gaoxiang12/slam_in_autonomous_driving)
- [optimized_ICP](https://github.com/zm0612/optimized_ICP)
- [small_gicp](https://github.com/koide3/small_gicp)
- [ndt_omp](https://github.com/koide3/ndt_omp)