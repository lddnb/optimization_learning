/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-19 10:42:55
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-23 18:51:40
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
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include "small_gicp/factors/icp_factor.hpp"
#include "small_gicp/factors/plane_icp_factor.hpp"

#include "optimization_learning/icp.hpp"
#include "optimization_learning/point_to_plane_icp.hpp"

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

  // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  // viewer->setBackgroundColor(0, 0, 0);
  // int v1(0);
  // int v2(1);
  // viewer->addPointCloud<pcl::PointXYZI>(source_points, "source_points");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "source_points");  // 红色
  // viewer->addPointCloud<pcl::PointXYZI>(target_points, "target_points");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "target_points");  // 绿色
  // viewer->addPointCloud<pcl::PointXYZI>(source_points_transformed, "source_points_transformed");
  // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "source_points_transformed");  // 蓝色

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

  LOG(INFO) << "------------------- GN ------------------";
  start = std::chrono::high_resolution_clock::now();
  Eigen::Affine3d T_opt = T_init;
  MatchP2P<pcl::PointXYZI>(source_points, T_init.matrix(), target_points, T_opt);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "------------------- PCL ICP ------------------";
  start = std::chrono::high_resolution_clock::now();
  pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
  icp.setInputSource(source_points);
  icp.setInputTarget(target_points);
  icp.setMaxCorrespondenceDistance(0.5);
  icp.setTransformationEpsilon(1e-5);
  icp.setEuclideanFitnessEpsilon(1e-5);
  icp.setMaximumIterations(30);

  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>);
  icp.align(*aligned, T_init.matrix().cast<float>());
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  Eigen::Affine3f T_icp(icp.getFinalTransformation());
  LOG(INFO) << "iterations: " << icp.nr_iterations_;
  LOG(INFO) << "R_opt: " << Eigen::Quaternionf(T_icp.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_icp.translation().transpose();

  LOG(INFO) << "------------------- small_gicp icp ------------------";
  start = std::chrono::high_resolution_clock::now();

  T_opt = T_init;
  ICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "======================== Point to Plane ICP ========================";
  PointToPlaneICPConfig config;


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

      std::vector<Eigen::Vector3d> plane_points;
      for (size_t i = 0; i < 5; ++i) {
        plane_points.emplace_back(
          target_points->at(nn_indices[i]).x,
          target_points->at(nn_indices[i]).y,
          target_points->at(nn_indices[i]).z);
      }
      Eigen::Matrix<double, 4, 1> plane_coeffs;
      if (nn_distances[0] > 1 || !FitPlane(plane_points, plane_coeffs)) {
        continue;
      }

      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CeresCostFunctorP2Plane, 1, 7>(
        new CeresCostFunctorP2Plane(source_points->at(i), target_points->at(nn_indices[0]), plane_coeffs.head<3>()));
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
    // todo：收敛阈值过小时会在两个点之间来回迭代，到达最大迭代次数后退出，原因未知
    if ((R.coeffs() - last_R.coeffs()).norm() < 1e-3 && (t - last_t).norm() < 1e-3) {
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

  LOG(INFO) << "------------------- GTSAM SE3 ------------------";
  gtsam::SharedNoiseModel noise_model2 = gtsam::noiseModel::Isotropic::Sigma(1, 1);
  gtsam::GaussNewtonParams params_gn2;
  // params_gn2.setVerbosity("ERROR");
  params_gn2.maxIterations = 1;
  params_gn2.relativeErrorTol = 1e-5;
  start = std::chrono::high_resolution_clock::now();
  last_T_gtsam = gtsam::Pose3(gtsam::Rot3(R_init), gtsam::Point3(t_init));
  for (iterations = 0; iterations < 50; ++iterations) {
    Eigen::Affine3d T_opt(last_T_gtsam.matrix());
    pcl::transformPointCloud(*source_points, *source_points_transformed, T_opt);

    gtsam::NonlinearFactorGraph graph;
    for (int i = 0; i < source_points->size(); ++i) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_distances(1);
      kdtree.nearestKSearch(source_points_transformed->at(i), 5, nn_indices, nn_distances);

      std::vector<Eigen::Vector3d> plane_points;
      for (size_t i = 0; i < 5; ++i) {
        plane_points.emplace_back(
          target_points->at(nn_indices[i]).x,
          target_points->at(nn_indices[i]).y,
          target_points->at(nn_indices[i]).z);
      }
      Eigen::Matrix<double, 4, 1> plane_coeffs;
       if (nn_distances[0] > 1 || !FitPlane(plane_points, plane_coeffs)) {
        continue;
      }

      graph.emplace_shared<GtsamIcpFactorP2Plane>(key, source_points->at(i), target_points->at(nn_indices[0]), plane_coeffs.head<3>(), noise_model2);
    }

    gtsam::Values initial_estimate;
    initial_estimate.insert(key, last_T_gtsam);
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn2);
    optimizer.optimize();

    result = optimizer.values();
    gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key);

    if (
      (last_T_gtsam.rotation().toQuaternion().coeffs() - T_result.rotation().toQuaternion().coeffs()).norm() < 1e-3 &&
      (last_T_gtsam.translation() - T_result.translation()).norm() < 1e-3) {
      break;
    }
    last_T_gtsam = T_result;
  }

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "GTSAM SE3 solve time: " << duration << " us";

  T_result = result.at<gtsam::Pose3>(key);
  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R: " << T_result.rotation().toQuaternion();
  LOG(INFO) << "t: " << T_result.translation().transpose();

  LOG(INFO) << "------------------- GTSAM SO3+R3 ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_GTSAM_SO3_R3<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "------------------- GN ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_GN<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "------------------- PCL NICP ------------------";
  start = std::chrono::high_resolution_clock::now();
  T_opt = T_init;
  P2PlaneICP_PCL<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "------------------- small_gicp  point to plane icp ------------------";
  start = std::chrono::high_resolution_clock::now();
  config.rotation_eps = 0.1 * M_PI / 180.0;
  T_opt = T_init;
  P2PlaneICP_small_gicp<pcl::PointXYZI>(source_points, target_points, T_opt, iterations, config);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Time elapsed: " << duration << " us";

  LOG(INFO) << "iterations: " << iterations;
  LOG(INFO) << "R_opt: " << Eigen::Quaterniond(T_opt.rotation()).coeffs().transpose();
  LOG(INFO) << "t_opt: " << T_opt.translation().transpose();

  LOG(INFO) << "======================== Generalized ICP ========================";
  return 0;
}