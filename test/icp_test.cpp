#include <gtest/gtest.h>
#include <optimization_learning/icp.hpp>

std::vector<Eigen::Vector3d> source_points;
std::vector<Eigen::Vector3d> target_points;

const Eigen::Quaterniond R_ture = Eigen::Quaterniond(Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitX()));
const Eigen::Vector3d t_ture = Eigen::Vector3d(1, 2, 3);

const int num_points = 10;        // 生成10个点
const double min_value = -100.0;  // 最小值
const double max_value = 100.0;   // 最大值

// 创建随机数生成器
std::random_device rd;
std::default_random_engine generator(rd());
std::uniform_real_distribution<double> distribution(min_value, max_value);

// initial guess
const Eigen::Quaterniond R_init = Eigen::Quaterniond(Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitX()));
const Eigen::Vector3d t_init = Eigen::Vector3d(2, 3, 4);

ceres::Solver::Options options;
gtsam::GaussNewtonParams params_gn;

TEST(ICPTest, ceres_SE3)
{
  double T[7] = {R_init.x(), R_init.y(), R_init.z(), R_init.w(), t_init.x(), t_init.y(), t_init.z()};

  ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>* se3 =
    new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
  ceres::Problem problem;
  problem.AddParameterBlock(T, 7, se3);

  for (int i = 0; i < num_points; i++) {
    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CeresCostFunctor, 3, 7>(new CeresCostFunctor(source_points[i], target_points[i]));
    problem.AddResidualBlock(cost_function, nullptr, T);
  }

  ceres::Solver::Summary summary;
  auto start = std::chrono::high_resolution_clock::now();
  ceres::Solve(options, &problem, &summary);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Ceres 1 solve time: " << duration << " us";

  // LOG(INFO) << summary.FullReport();

  Eigen::Map<Eigen::Quaterniond> R(T);
  Eigen::Map<Eigen::Vector3d> t(T + 4);

  LOG(INFO) << "----------- Ceres 1 -----------";
  LOG(INFO) << "R: " << R.coeffs().transpose();
  LOG(INFO) << "t: " << t.transpose();
  EXPECT_NEAR((R.inverse() * R_ture).norm(), 1.0, 1e-6);
  EXPECT_NEAR((t - t_ture).norm(), 0.0, 1e-6);
}

TEST(ICPTest, ceres_SO3_R3)
{
  double T_R[4] = {R_init.x(), R_init.y(), R_init.z(), R_init.w()};
  double T_t[3] = {t_init.x(), t_init.y(), t_init.z()};
  ceres::Manifold* so3 = new RightQuaternionManifold(); // ceres::EigenQuaternionManifold
  ceres::Problem problem_2;
  problem_2.AddParameterBlock(T_R, 4, so3);
  problem_2.AddParameterBlock(T_t, 3);
  for (int i = 0; i < num_points; i++) {
    ceres::CostFunction* cost_function =
      new MyCossFunction(source_points[i], target_points[i]);
    problem_2.AddResidualBlock(cost_function, nullptr, T_R, T_t);
  }
  ceres::Solver::Summary summary_2;
  auto start = std::chrono::high_resolution_clock::now();
  ceres::Solve(options, &problem_2, &summary_2);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Ceres 2 solve time: " << duration << " us";

  Eigen::Map<Eigen::Quaterniond> R2(T_R);
  Eigen::Map<Eigen::Vector3d> t2(T_t);

  LOG(INFO) << "----------- Ceres 2 -----------";
  LOG(INFO) << "R: " << R2.coeffs().transpose();
  LOG(INFO) << "t: " << t2.transpose();
  EXPECT_NEAR((R2.inverse() * R_ture).norm(), 1.0, 1e-1);
  EXPECT_NEAR((t2 - t_ture).norm(), 0.0, 1e-1);
}

TEST(ICPTest, SVD)
{
  LOG(INFO) << "----------- SVD -----------";
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d source_bar = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_bar = Eigen::Vector3d::Zero();

  source_bar = std::accumulate(source_points.begin(), source_points.end(), source_bar) / source_points.size();
  target_bar = std::accumulate(target_points.begin(), target_points.end(), target_bar) / target_points.size();

  // LOG(INFO) << "source_bar: " << source_bar.transpose();
  // LOG(INFO) << "target_bar: " << target_bar.transpose();

  for (int i = 0; i < num_points; i++) {
    A += (source_points[i] - source_bar) * (target_points[i] - target_bar).transpose();
  }
  LOG(INFO) << "det(A): " << A.determinant();
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "SVD solve time: " << duration << " us";

  Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
  Eigen::Vector3d t_svd = target_bar - R_svd * source_bar;

  LOG(INFO) << "det(R): " << R_svd.determinant();
  LOG(INFO) << "R: " << Eigen::Quaterniond(R_svd).coeffs().transpose();
  LOG(INFO) << "t: " << t_svd.transpose();

  EXPECT_NEAR((R_svd.transpose() * R_ture).squaredNorm(), 3.0, 1e-6);
  EXPECT_NEAR((t_svd - t_ture).norm(), 0.0, 1e-6);
}

TEST(ICPTest, GTSAM_SE3)
{
  LOG(INFO) << "----------- GTSAM SE3 -----------";
  const gtsam::Pose3 T_true = gtsam::Pose3(gtsam::Rot3(R_ture), gtsam::Point3(t_ture));
  std::vector<gtsam::Point3> source_gtsam(source_points.begin(), source_points.end());
  std::vector<gtsam::Point3> target_gtsam(target_points.begin(), target_points.end());
  // gtsam::noiseModel::Isotropic cost_model = gtsam::noiseModel::Isotropic(3, 1.0);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
  gtsam::NonlinearFactorGraph graph;
  for (int i = 0; i < num_points; i++) {
    graph.emplace_shared<GtsamIcpFactor>(key, source_gtsam[i], target_gtsam[i], noise_model);
  }
  gtsam::Values initial_estimate;
  gtsam::Pose3 T_init = gtsam::Pose3(gtsam::Rot3(R_init), gtsam::Point3(t_init));
  initial_estimate.insert(key, T_init);
  gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate, params_gn);

  auto start = std::chrono::high_resolution_clock::now();
  optimizer.optimize();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "GTSAM SE3 solve time: " << duration << " us";

  gtsam::Values result = optimizer.values();
  gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key);

  LOG(INFO) << "R: " << T_result.rotation().toQuaternion();
  LOG(INFO) << "t: " << T_result.translation().transpose();

  EXPECT_NEAR((T_result.rotation().inverse() * gtsam::Rot3(R_ture)).toQuaternion().norm(), 1.0, 1e-6);
  EXPECT_NEAR((T_result.translation() - t_ture).norm(), 0.0, 1e-6);
}

TEST(ICPTest, GTSAM_SO3_R3)
{
  LOG(INFO) << "----------- GTSAM SO3+R3 -----------";
  const gtsam::Key key = gtsam::symbol_shorthand::X(0);
  const gtsam::Key key2 = gtsam::symbol_shorthand::X(1);
  gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 1);
  std::vector<gtsam::Point3> source_gtsam(source_points.begin(), source_points.end());
  std::vector<gtsam::Point3> target_gtsam(target_points.begin(), target_points.end());
  gtsam::NonlinearFactorGraph graph2;
  for (int i = 0; i < num_points; i++) {
    graph2.emplace_shared<GtsamIcpFactor2>(key, key2, source_gtsam[i], target_gtsam[i], noise_model);
  }
  gtsam::Values initial_estimate2;
  initial_estimate2.insert(key, gtsam::Rot3(R_init));
  initial_estimate2.insert(key2, gtsam::Point3(t_init));
  gtsam::GaussNewtonOptimizer optimizer2(graph2, initial_estimate2, params_gn);

  auto start = std::chrono::high_resolution_clock::now();
  optimizer2.optimize();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "GTSAM SO3+R3 solve time: " << duration << " us";

  gtsam::Values result2 = optimizer2.values();
  gtsam::Rot3 R_result = result2.at<gtsam::Rot3>(key);
  gtsam::Point3 t_result = result2.at<gtsam::Point3>(key2);

  LOG(INFO) << "R: " << R_result.toQuaternion();
  LOG(INFO) << "t: " << t_result.transpose();

  EXPECT_NEAR((R_result.inverse() * gtsam::Rot3(R_ture)).toQuaternion().norm(), 1.0, 1e-6);
  EXPECT_NEAR((t_result - t_ture).norm(), 0.0, 1e-6);
}

TEST(ICPTest, Gauss_Newton)
{
  LOG(INFO) << "----------- Gauss-Newton -----------";
  Eigen::Matrix<double, 6, 1> delta;
  delta << 1e5, 1e5, 1e5, 1e5, 1e5, 1e5;
  Eigen::Matrix3d R_iter = R_init.toRotationMatrix();
  Eigen::Vector3d t_iter = t_init;
  int iter = 0;

  auto start = std::chrono::high_resolution_clock::now();
  while (delta.norm() > 1e-5 && iter < 10) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();  //
    Eigen::Matrix<double, 6, 1> B = Eigen::Matrix<double, 6, 1>::Zero();  //
    for (int i = 0; i < num_points; i++) {
      Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
      J.leftCols(3) = -R_iter * Hat(source_points[i]);
      J.rightCols(3) = Eigen::Matrix<double, 3, 3>::Identity();
      H += J.transpose() * J;
      B += -J.transpose() * (R_iter * source_points[i] + t_iter - target_points[i]);
    }
    if (H.determinant() < 1e-5) {
      LOG(INFO) << "H is singular, cannot compute delta.";
      continue;
    }
    delta = H.inverse() * B;
    R_iter *= Exp(delta.head<3>());
    t_iter += delta.tail<3>();
    // LOG(INFO) << "[" << iter++ << "] delta: " << delta.transpose() << ", t: " << t_iter.transpose();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  LOG(INFO) << "Gauss-Newton solve time: " << duration << " us";

  LOG(INFO) << "R: " << Eigen::Quaterniond(R_iter).coeffs().transpose();
  LOG(INFO) << "t: " << t_iter.transpose();

  EXPECT_NEAR((R_iter.transpose() * R_ture).squaredNorm(), 3.0, 1e-6);
  EXPECT_NEAR((t_iter - t_ture).norm(), 0.0, 1e-6);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  google::InstallFailureSignalHandler();  // 配置安装程序崩溃失败信号处理器
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;

  // 生成随机点
  for (int i = 0; i < num_points; ++i) {
    double x = distribution(generator);  // x
    double y = distribution(generator);  // y
    double z = distribution(generator);  // z
    source_points.emplace_back(Eigen::Vector3d(x, y, z));
    target_points.emplace_back(R_ture * source_points.back() + t_ture);
  }

  LOG(INFO) << "R_init: " << R_init.coeffs().transpose();
  LOG(INFO) << "t_init: " << t_init.transpose();
  LOG(INFO) << "R_Ture: " << R_ture.coeffs().transpose();
  LOG(INFO) << "t_ture: " << t_ture.transpose();

  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  // params_gn.setVerbosity("ERROR");
  params_gn.maxIterations = 100;
  params_gn.relativeErrorTol = 1e-5;

  return RUN_ALL_TESTS();
}