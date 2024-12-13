/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-09 11:56:59
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-13 14:17:20
 * @ Description:
 */

#include <random>
#include <glog/logging.h>

#include "optimization_learning/R_mean.hpp"

int main(int argc, char** argv)
{
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  const int num_points = 10;     // 生成10个点
  const double min_value = 1.0;  // 最小值
  const double max_value = 2.0;  // 最大值

  // 创建随机数生成器
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<double> distribution(min_value, max_value);

  // 生成随机点
  std::vector<Eigen::Quaterniond> R;
  for (int i = 0; i < num_points; ++i) {
    double x = distribution(generator);  // x
    double y = distribution(generator);  // y
    double z = distribution(generator);  // z
    Eigen::Vector3d axis(x, y, z);
    axis.normalize();
    R.emplace_back(Eigen::AngleAxisd(distribution(generator) * M_PI_4, axis));
    R.back().normalize();
  }

  // 一、Ceres 第一种残差计算方式
  // 从结果来看，这个Log计算方式收敛更快一些
  LOG(INFO) << "----------- Ceres::QuaternionToAngleAxis test -----------";

  Eigen::Quaterniond R_res = R[0];
  Eigen::Quaterniond R_init = R[0];

  ceres::Problem problem_1;
  ceres::Manifold* quaternion_manifold_1 = new ceres::EigenQuaternionManifold(); // ceres::EigenQuaternionManifold() RightQuaternionManifold

  for (auto& r : R) {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor1, 3, 4>(new CostFunctor1(r));
    problem_1.AddResidualBlock(cost_function, NULL, R_res.coeffs().data());
    problem_1.SetManifold(R_res.coeffs().data(), quaternion_manifold_1);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_1, &summary);

  // LOG(INFO) << summary.FullReport();

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> " << R_res.coeffs().transpose();

  // 二、Ceres 第二种残差计算方式
  LOG(INFO) << "----------- Eigen::AngleAxis test -----------";
  R_res = R[0];
  ceres::Problem problem_2;
  ceres::Manifold* quaternion_manifold_2 = new RightQuaternionManifold; // ceres::EigenQuaternionManifold();

  for (auto& r : R) {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor2, 3, 4>(new CostFunctor2(r));
    problem_2.AddResidualBlock(cost_function, NULL, R_res.coeffs().data());
    problem_2.SetManifold(R_res.coeffs().data(), quaternion_manifold_2);
  }

  ceres::Solver::Summary summary_2;
  ceres::Solve(options, &problem_2, &summary_2);

  // LOG(INFO) << summary_2.FullReport();

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> " << R_res.coeffs().transpose();

  // 三、Ceres 手动输入雅克比求解
  LOG(INFO) << "----------- Ceres input Jacobian test -----------";
  
  R_res = R[0];
  ceres::Problem problem_3;
  ceres::Manifold* quaternion_manifold_3 = new ceres::EigenQuaternionManifold(); // ceres::EigenQuaternionManifold();

  for (auto& r : R) {
    ceres::CostFunction* cost_function = new MyCostFunction(r);
    problem_3.AddResidualBlock(cost_function, NULL, R_res.coeffs().data());
    problem_3.SetManifold(R_res.coeffs().data(), quaternion_manifold_2);
  }

  ceres::Solver::Summary summary_3;
  ceres::Solve(options, &problem_3, &summary_3);

  // LOG(INFO) << summary_3.FullReport();

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> " << R_res.coeffs().transpose();

  // 四、GTSAM 优化
  LOG(INFO) << "----------- GTSAM test -----------";
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
  gtsam::NonlinearFactorGraph graph;
  for (auto& r : R) {
    graph.add(GtsamFactor(key, r, gtsam::noiseModel::Isotropic::Sigma(3, 1)));
  }
  gtsam::Values initial_estimate;
  gtsam::Rot3 R_init_gtsam = gtsam::Rot3(R_init);
  initial_estimate.insert(key, R_init_gtsam);
  gtsam::GaussNewtonParams params_gn;
  params_gn.maxIterations = 100;
  params_gn.relativeErrorTol = 1e-5;
  gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate);
  optimizer.optimize();
  gtsam::Values result = optimizer.values();
  gtsam::Rot3 T_result = result.at<gtsam::Rot3>(key);

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> "
            << Eigen::Quaterniond(T_result.matrix()).coeffs().transpose();

  // 五、GTSAM 自带的先验因子优化
  LOG(INFO) << "----------- GTSAM prior test -----------";
  gtsam::NonlinearFactorGraph graph_prior;
  for (auto& r : R) {
    graph_prior.add(gtsam::PriorFactor<gtsam::Rot3>(key, gtsam::Rot3(r), gtsam::noiseModel::Isotropic::Sigma(3, 1)));
  }
  gtsam::GaussNewtonOptimizer optimizer_prior(graph_prior, initial_estimate);
  optimizer_prior.optimize();
  gtsam::Values result_prior = optimizer_prior.values();
  gtsam::Rot3 T_result_prior = result_prior.at<gtsam::Rot3>(key);

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> "
            << Eigen::Quaterniond(T_result_prior.matrix()).coeffs().transpose();

  // 六、Eigen 和 GTSAM 中旋转的各种表示和Log计算
  LOG(INFO) << "----------- Eigen and GTSAM Rotation -----------";
  // 注意旋转的归一化，保证创建的矩阵是旋转矩阵，或者四元数是单位四元数
  // 使用 AngleAxisd 构建旋转时要对旋转向量归一化 Eigen::Vector3d().normalize
  Eigen::Quaterniond tmp_R = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d(1, 2, 3).normalized()));  //
  LOG(INFO) << "!!! Eigen !!!";
  LOG(INFO) << "R: " << std::endl << tmp_R.toRotationMatrix();
  LOG(INFO) << "q(xyzw): " << tmp_R.coeffs().transpose();
  auto vec = Eigen::AngleAxisd(tmp_R);
  LOG(INFO) << "AngleAxisd: " << vec.angle() * vec.axis().transpose();
  LOG(INFO) << "angle: " << vec.angle() / M_PI * 180;
  LOG(INFO) << "axis: " << vec.axis().transpose();

  LOG(INFO) << "!!! Ceres !!!";
  Eigen::Vector3d vec_2 = Eigen::Vector3d::Zero();
  Eigen::Vector4d ceres_q{tmp_R.w(), tmp_R.x(), tmp_R.y(), tmp_R.z()};
  ceres::QuaternionToAngleAxis(ceres_q.data(), vec_2.data());
  LOG(INFO) << "AngleAxisd: " << vec_2.transpose();
  LOG(INFO) << "!!! GTSAM !!!";
  auto R_gtsam = gtsam::Rot3(tmp_R);
  LOG(INFO) << "R: " << std::endl << R_gtsam.matrix();
  LOG(INFO) << "q(wxyz): " << R_gtsam.quaternion().transpose();
  LOG(INFO) << "angle: " << R_gtsam.axisAngle().second / M_PI * 180;

  gtsam::Vector axis_gtsam(R_gtsam.axisAngle().first.point3());
  LOG(INFO) << "axis: " << axis_gtsam.transpose();
  LOG(INFO) << "AngleAxisd: " << (axis_gtsam * R_gtsam.axisAngle().second).transpose();
  LOG(INFO) << "Logmap: " << gtsam::Rot3::Logmap(R_gtsam).transpose();

  return 0;
}