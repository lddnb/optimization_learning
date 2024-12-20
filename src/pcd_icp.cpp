/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-19 10:42:55
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-20 10:40:58
 * @ Description:
 */

#include <execution>
#include <rclcpp/rclcpp.hpp>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"


// https://github.com/zm0612/optimized_ICP/blob/be8651addd630c472418cf530a53623946906831/optimized_ICP_GN.cpp#L17
bool Match(
  const pcl::PointCloud<pcl::PointXYZI>::Ptr& source_cloud_ptr,
  const Eigen::Matrix4d& predict_pose,
  const pcl::PointCloud<pcl::PointXYZI>::Ptr& target_cloud_ptr,
  Eigen::Affine3d& result_pose)
{
  bool has_converge_ = false;
  int max_iterations_ = 30;
  double max_correspond_distance_ = 0.5;
  double transformation_epsilon_ = 1e-5;

  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_flann_ptr_(new pcl::KdTreeFLANN<pcl::PointXYZI>());
  kdtree_flann_ptr_->setInputCloud(target_cloud_ptr);

  Eigen::Matrix4d T = predict_pose;

  // Gauss-Newton's method solve ICP.
  unsigned int i = 0;
  for (; i < max_iterations_; ++i) {
    pcl::transformPointCloud(*source_cloud_ptr, *transformed_cloud, T);
    Eigen::Matrix<double, 6, 6> Hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> B = Eigen::Matrix<double, 6, 1>::Zero();

    for (unsigned int j = 0; j < transformed_cloud->size(); ++j) {
      const pcl::PointXYZI& origin_point = source_cloud_ptr->points[j];

      // 删除距离为无穷点
      if (!pcl::isFinite(origin_point)) {
        continue;
      }

      const pcl::PointXYZI& transformed_point = transformed_cloud->at(j);
      std::vector<float> resultant_distances;
      std::vector<int> indices;
      // 在目标点云中搜索距离当前点最近的一个点
      kdtree_flann_ptr_->nearestKSearch(transformed_point, 1, indices, resultant_distances);

      // 舍弃那些最近点,但是距离大于最大对应点对距离
      if (resultant_distances.front() > max_correspond_distance_) {
        continue;
      }

      Eigen::Vector3d nearest_point = Eigen::Vector3d(
        target_cloud_ptr->at(indices.front()).x,
        target_cloud_ptr->at(indices.front()).y,
        target_cloud_ptr->at(indices.front()).z);

      Eigen::Vector3d point_eigen(transformed_point.x, transformed_point.y, transformed_point.z);
      Eigen::Vector3d origin_point_eigen(origin_point.x, origin_point.y, origin_point.z);
      Eigen::Vector3d error = point_eigen - nearest_point;

      Eigen::Matrix<double, 3, 6> Jacobian = Eigen::Matrix<double, 3, 6>::Zero();
      // 构建雅克比矩阵
      Jacobian.leftCols(3) = Eigen::Matrix3d::Identity();
      Jacobian.rightCols(3) = -T.block<3, 3>(0, 0) * Hat(origin_point_eigen);

      // 构建海森矩阵
      Hessian += Jacobian.transpose() * Jacobian;
      B += -Jacobian.transpose() * error;
    }

    if (Hessian.determinant() == 0) {
      continue;
    }

    Eigen::Matrix<double, 6, 1> delta_x = Hessian.inverse() * B;

    T.block<3, 1>(0, 3) = T.block<3, 1>(0, 3) + delta_x.head(3);
    T.block<3, 3>(0, 0) *= Exp(delta_x.tail(3)).matrix();

    if (delta_x.norm() < transformation_epsilon_) {
      has_converge_ = true;
      break;
    }

    // debug
    // LOG(INFO) << "i= " << i << "  norm delta x= " << delta_x.norm();
  }
  LOG(INFO) << "iterations: " << i;

  result_pose = T;

  return true;
}

bool next_iteration = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
  if (event.getKeySym() == "space" && event.keyDown()) next_iteration = true;
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("pcd_icp");

  std::string pcd_file_path = "/home/ubuntu/ros_ws/src/optimization_learning/data/";
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_points(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_points(new pcl::PointCloud<pcl::PointXYZI>);
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
  
  const Eigen::Quaterniond R_ture = Eigen::Quaterniond(Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitX()));
  const Eigen::Vector3d t_ture = Eigen::Vector3d(1, 2, 3);
  const Eigen::Affine3d T_true(Eigen::Translation3d(t_ture) * R_ture.toRotationMatrix());
  pcl::transformPointCloud(*target_points, *target_points, T_true);

  const Eigen::Quaterniond R_init = Eigen::Quaterniond(Eigen::AngleAxisd(1.45, Eigen::Vector3d::UnitX()));
  const Eigen::Vector3d t_init = Eigen::Vector3d(1.2, 2.2, 3.2);
  const Eigen::Affine3d T_init(Eigen::Translation3d(t_init) * R_init.toRotationMatrix());
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_points_transformed(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*source_points, *source_points_transformed, T_init);

  LOG(INFO) << "R_init: " << R_init.coeffs().transpose();
  LOG(INFO) << "t_init: " << t_init.transpose();
  LOG(INFO) << "R_Ture: " << R_ture.coeffs().transpose();
  LOG(INFO) << "t_ture: " << t_ture.transpose();

  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  int v1(0);
  int v2(1);
  viewer->addPointCloud<pcl::PointXYZI>(source_points, "source_points");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_points");  // 红色
  viewer->addPointCloud<pcl::PointXYZI>(target_points, "target_points");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "target_points");  // 绿色
  viewer->addPointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "source_points_transformed");  // 蓝色

  // viewer->spin();

  // 逐步可视化
  // viewer->registerKeyboardCallback (&keyboardEventOccurred, nullptr);
  // while (!viewer->wasStopped ()) {
  // viewer->spinOnce();
  // if (next_iteration) {
  //   // ...
  //   viewer->updatePointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
  // }
  // next_iteration = false;

  // 创建kd树
  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree = pcl::KdTreeFLANN<pcl::PointXYZI>();
  kdtree.setInputCloud(target_points);

  LOG(INFO) << "======================== Point to Point ICP ========================";

  LOG(INFO) << "------------------- Ceres ------------------";
  std::vector<double> T = {R_init.x(), R_init.y(), R_init.z(), R_init.w(), t_init.x(), t_init.y(), t_init.z()};

  int iterations = 0;
  Eigen::Quaterniond last_R = R_init;
  Eigen::Vector3d last_t = t_init;
  auto start = std::chrono::high_resolution_clock::now();
  for (iterations = 0; iterations < 50; ++iterations) {
    Eigen::Affine3d T_opt(Eigen::Translation3d(last_t) * last_R.toRotationMatrix());
    pcl::transformPointCloud(*source_points, *source_points_transformed, T_opt);

    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    ceres::Problem problem;
    problem.AddParameterBlock(T.data(), 7, se3);

    for (int i = 0; i < source_points->size(); i++) {
      // 最近邻搜索
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree.nearestKSearch(source_points_transformed->at(i), 1, nn_indices, nn_distances);

      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctor, 3, 7>(
        new CeresCostFunctor(source_points->at(i), target_points->at(nn_indices[0])));
      problem.AddResidualBlock(cost_function, nullptr, T.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Map<Eigen::Quaterniond> R(T.data());
    Eigen::Map<Eigen::Vector3d> t(T.data() + 4);
    if ((R.coeffs() - last_R.coeffs()).norm() < 1e-5 && (t - last_t).norm() < 1e-5) {
      break;
    }
    last_R = R;
    last_t = t;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  Eigen::Map<Eigen::Quaterniond> R(T.data());
  Eigen::Map<Eigen::Vector3d> t(T.data() + 4);

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << R.coeffs().transpose();
  LOG(INFO) << "t: " << t.transpose();

  LOG(INFO) << "------------------- GTSAM SO3+R3 ------------------";
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
  const gtsam::Key key2 = gtsam::symbol_shorthand::X(1);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);

  // 将 pcl 点云转换为 gtsam 点云
  // gtsam::Point3Vector source_gtsam(source_points->size());
  // gtsam::Point3Vector target_gtsam(target_points->size());
  // std::transform(
  //   std::execution::par,
  //   source_points->begin(),
  //   source_points->end(),
  //   source_gtsam.begin(),
  //   [](const pcl::PointXYZI& point) { return gtsam::Point3(point.x, point.y, point.z); });
  // std::transform(
  //   std::execution::par,
  //   target_points->begin(),
  //   target_points->end(),
  //   target_gtsam.begin(),
  //   [](const pcl::PointXYZI& point) { return gtsam::Point3(point.x, point.y, point.z); });

  gtsam::Values result;

  start = std::chrono::high_resolution_clock::now();
  gtsam::Rot3 last_R_gtsam = gtsam::Rot3(R_init);
  gtsam::Point3 last_t_gtsam = gtsam::Point3(t_init);
  gtsam::GaussNewtonParams params_gn;
  // params_gn.setVerbosity("ERROR");
  params_gn.maxIterations = 1;
  params_gn.relativeErrorTol = 1e-5;
  for (iterations = 0; iterations < 50; ++iterations) {
    Eigen::Affine3d T_opt(Eigen::Translation3d(last_t_gtsam) * last_R_gtsam.matrix());
    pcl::transformPointCloud(*source_points, *source_points_transformed, T_opt);

    gtsam::NonlinearFactorGraph graph;
    for (int i = 0; i < source_points->size(); ++i) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree.nearestKSearch(source_points_transformed->at(i), 1, nn_indices, nn_distances);
      graph.emplace_shared<GtsamIcpFactor2>(key, key2, source_points->at(i), target_points->at(nn_indices[0]), noise_model);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key, last_R_gtsam);
    initial_estimate.insert(key2, last_t_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);

    optimizer.optimize();

    result = optimizer.values();
    gtsam::Rot3 R_result = result.at<gtsam::Rot3>(key);
    gtsam::Point3 t_result = result.at<gtsam::Point3>(key2);

    if (
      (R_result.toQuaternion().coeffs() - last_R_gtsam.toQuaternion().coeffs()).norm() < 1e-5 &&
      (t_result - last_t_gtsam).norm() < 1e-5) {
      break;
    }
    last_R_gtsam = R_result;
    last_t_gtsam = t_result;
  }

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "GTSAM SO3+R3 solve time: " << duration << " us";

  gtsam::Rot3 R_result = result.at<gtsam::Rot3>(key);
  gtsam::Point3 t_result = result.at<gtsam::Point3>(key2);
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << R_result.toQuaternion();
  LOG(INFO) << "t: " << t_result.transpose();

  LOG(INFO) << "------------------- GTSAM SE3 ------------------";
  start = std::chrono::high_resolution_clock::now();
  gtsam::Pose3 last_T_gtsam = gtsam::Pose3(gtsam::Rot3(R_init), gtsam::Point3(t_init));
  for (iterations = 0; iterations < 50; ++iterations) {
    Eigen::Affine3d T_opt(last_T_gtsam.matrix());
    pcl::transformPointCloud(*source_points, *source_points_transformed, T_opt);

    gtsam::NonlinearFactorGraph graph;
    for (int i = 0; i < source_points->size(); ++i) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree.nearestKSearch(source_points_transformed->at(i), 1, nn_indices, nn_distances);
      graph.emplace_shared<GtsamIcpFactor>(key, source_points->at(i), target_points->at(nn_indices[0]), noise_model);
    }

    gtsam::Values initial_estimate2;
    initial_estimate2.insert(key, last_T_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate2, params_gn);
    optimizer.optimize();

    result = optimizer.values();
    gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key);

    if (
      (last_T_gtsam.rotation().toQuaternion().coeffs() - T_result.rotation().toQuaternion().coeffs()).norm() < 1e-5 &&
      (last_T_gtsam.translation() - T_result.translation()).norm() < 1e-5) {
      break;
    }
    last_T_gtsam = T_result;
  }

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "GTSAM SE3 solve time: " << duration << " us";

  gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key);
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << T_result.rotation().toQuaternion();
  LOG(INFO) << "t: " << T_result.translation().transpose();

  // LOG(INFO) << "------------------- STD ------------------";
  // // 计算法向量
  // pcl::PointCloud<pcl::PointXYZINormal>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  // pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
  // pcl::copyPointCloud(*source_points, *source_cloud);
  // pcl::copyPointCloud(*target_points, *target_cloud);

  // pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> norm_est;
  // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  // pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
  // norm_est.setKSearch(30);
  // norm_est.setNumberOfThreads(10);
  // norm_est.setSearchMethod(tree);
  // norm_est.setInputCloud(source_points);
  // norm_est.compute(*normals);

  // for (int i = 0; i < source_cloud->size(); ++i) {
  //   source_cloud->at(i).normal_x = normals->at(i).normal_x;
  //   source_cloud->at(i).normal_y = normals->at(i).normal_y;
  //   source_cloud->at(i).normal_z = normals->at(i).normal_z;
  // }

  // norm_est.setInputCloud(target_points);
  // norm_est.compute(*normals);

  // for (int i = 0; i < target_cloud->size(); ++i) {
  //   target_cloud->at(i).normal_x = normals->at(i).normal_x;
  //   target_cloud->at(i).normal_y = normals->at(i).normal_y;
  //   target_cloud->at(i).normal_z = normals->at(i).normal_z;
  // }
  
  // std::pair<Eigen::Vector3d, Eigen::Matrix3d> transform;
  // transform.first = t_init;
  // transform.second = R_init.toRotationMatrix();
  // PlaneGeomrtricIcp(source_cloud, target_cloud, transform);

  // LOG(INFO) << "R_opt: " << Eigen::Quaterniond(transform.second).coeffs().transpose();
  // LOG(INFO) << "t_opt: " << transform.first.transpose();


  LOG(INFO) << "------------------- GN ------------------";
  start = std::chrono::high_resolution_clock::now();
  Eigen::Affine3d T_opt = T_init;
  Match(source_points, T_init.matrix(), target_points, T_opt);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "------------------- ICP ------------------";
  start = std::chrono::high_resolution_clock::now();
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setInputSource(source_points);
  icp.setInputTarget(target_points);
  icp.setMaxCorrespondenceDistance(0.5);
  icp.setTransformationEpsilon(1e-5);
  icp.setEuclideanFitnessEpsilon(1e-5);
  icp.setMaximumIterations(30);

  icp.align(*source_points, T_init.matrix().cast<float>());
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  Eigen::Affine3f T_icp(icp.getFinalTransformation());
  LOG(INFO) << "iterations: " << icp.nr_iterations_;
  LOG(INFO) << "R_opt: " << Eigen::Quaternionf(T_icp.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_icp.translation().transpose();


  LOG(INFO) << "======================== Point to Plane ICP ========================";
  LOG(INFO) << "------------------- Ceres ------------------";
  T = {R_init.x(), R_init.y(), R_init.z(), R_init.w(), t_init.x(), t_init.y(), t_init.z()};

  last_R = R_init;
  last_t = t_init;
  start = std::chrono::high_resolution_clock::now();
  for (iterations = 0; iterations < 50; ++iterations) {
    Eigen::Affine3d T_opt(Eigen::Translation3d(last_t) * last_R.toRotationMatrix());
    pcl::transformPointCloud(*source_points, *source_points_transformed, T_opt);

    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
      new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    ceres::Problem problem;
    problem.AddParameterBlock(T.data(), 7, se3);

    for (int i = 0; i < source_points->size(); i++) {
      // 最近邻搜索
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree.nearestKSearch(source_points_transformed->at(i), 5, nn_indices, nn_distances);

      if (nn_distances[0] > 0.5) continue;

      
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctor, 3, 7>(
        new CeresCostFunctor(source_points->at(i), target_points->at(nn_indices[0])));
      problem.AddResidualBlock(cost_function, nullptr, T.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Map<Eigen::Quaterniond> R(T.data());
    Eigen::Map<Eigen::Vector3d> t(T.data() + 4);
    if ((R.coeffs() - last_R.coeffs()).norm() < 1e-5 && (t - last_t).norm() < 1e-5) {
      break;
    }
    last_R = R;
    last_t = t;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  Eigen::Map<Eigen::Quaterniond> R_p2p(T.data());
  Eigen::Map<Eigen::Vector3d> t_p2p(T.data() + 4);

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << R_p2p.coeffs().transpose();
  LOG(INFO) << "t: " << t_p2p.transpose();




  return 0;
}