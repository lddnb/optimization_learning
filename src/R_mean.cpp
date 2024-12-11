/**
 * @ Author: lddnb
 * @ Create Time: 2024-12-09 11:56:59
 * @ Modified by: lddnb
 * @ Modified time: 2024-12-09 16:41:19
 * @ Description:
 */

#include <random>
#include <glog/logging.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

// 四元数对数映射log，而非Log，中间差了一半
template <typename T>
Eigen::Matrix<T, 3, 1> log_Quaternion(const Eigen::Quaternion<T>& q)
{
  const T u_norm = ceres::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  const T theta = ceres::atan2(u_norm, q.w());

  Eigen::Matrix<T, 3, 1> ret = Eigen::Matrix<T, 3, 1>::Zero();
  if (ceres::fpclassify(u_norm) != FP_ZERO) {
    ret(0) = theta * q.x() / u_norm;
    ret(1) = theta * q.y() / u_norm;
    ret(2) = theta * q.z() / u_norm;
  }

  return ret;
}

// 旋转残差
class CostFunctor1
{
public:
  CostFunctor1(const Eigen::Matrix3d& R_) : R0(R_.transpose()) {}
  CostFunctor1(const Eigen::Quaterniond& R_) : R0(R_.inverse()) {}

  template <typename T>
  bool operator()(const T* const R_, T* residual_) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> R(R_);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residual_);

    // Eigen::Quaternion<T> q_err = R0.cast<T>().inverse() * R;
    Eigen::Quaternion<T> q_err = R0.cast<T>() * R;

    residual = log_Quaternion(q_err);

    return true;
  }

private:
  Eigen::Quaterniond R0;
};

class CostFunctor2
{
public:
  CostFunctor2(const Eigen::Matrix3d& R) : R0(R.transpose()) {}
  CostFunctor2(const Eigen::Quaterniond& R) : R0(R.inverse()) {}

  template <typename T>
  bool operator()(const T* const q, T* residual) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> q_eigen(q);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_eigen(residual);

    Eigen::Quaternion<T> R_delta = R0.cast<T>() * q_eigen;

    auto residual_vec = Eigen::AngleAxis<T>(R_delta);
    residual_eigen = residual_vec.angle() * residual_vec.axis() * 0.5;
    return true;
  }

private:
  Eigen::Quaterniond R0;
};

// ceres 手动求导
class MyCostFunction : public ceres::SizedCostFunction<3, 4>
{
public:
  MyCostFunction(const Eigen::Quaterniond& R0) : R0(R0.inverse()) {}

private:
  Eigen::Quaterniond R0;
};

class GtsamFactor : public gtsam::NoiseModelFactor1<gtsam::Rot3>
{
public:
  GtsamFactor(gtsam::Key key, const Eigen::Quaterniond& R, const gtsam::SharedNoiseModel& model)
  : gtsam::NoiseModelFactor1<gtsam::Rot3>(model, key),
    R0(R.inverse())
  {
  }

  virtual gtsam::Vector evaluateError(const gtsam::Rot3& R, boost::optional<gtsam::Matrix&> H = boost::none) const override
  {
    gtsam::Rot3 R_delta = R0 * R;
    gtsam::Vector residual = gtsam::Rot3::Logmap(R_delta) * 0.5;
    if (H) {
      *H = gtsam::I_3x3;
    }
    return residual;
  }

private:
  gtsam::Rot3 R0;
};

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
  LOG(INFO) << "----------- Ceres test 1 -----------";

  Eigen::Quaterniond R_res = R[0];
  Eigen::Quaterniond R_init = R[0];

  ceres::Problem problem_1;
  ceres::Manifold* quaternion_manifold_1 = new ceres::EigenQuaternionManifold();

  for (auto& r : R) {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor1, 3, 4>(new CostFunctor1(r));
    problem_1.AddResidualBlock(cost_function, NULL, R_res.coeffs().data());
    problem_1.SetManifold(R_res.coeffs().data(), quaternion_manifold_1);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_1, &summary);

  // LOG(INFO) << summary.FullReport();

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> " << R_res.coeffs().transpose();

  // 二、Ceres 第二种残差计算方式
  LOG(INFO) << "----------- Ceres test 2 -----------";
  R_res = R[0];
  ceres::Problem problem_2;
  ceres::Manifold* quaternion_manifold_2 = new ceres::EigenQuaternionManifold();

  for (auto& r : R) {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor2, 3, 4>(new CostFunctor2(r));
    problem_2.AddResidualBlock(cost_function, NULL, R_res.coeffs().data());
    problem_2.SetManifold(R_res.coeffs().data(), quaternion_manifold_2);
  }

  ceres::Solver::Options options_2;
  options_2.linear_solver_type = ceres::DENSE_QR;
  options_2.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary_2;
  ceres::Solve(options_2, &problem_2, &summary_2);

  // LOG(INFO) << summary_2.FullReport();

  LOG(INFO) << "R : " << R_init.coeffs().transpose() << " -> " << R_res.coeffs().transpose();

  // 三、GTSAM 优化
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

  // 四、GTSAM 自带的先验因子优化
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

  // 五、Eigen 和 GTSAM 中旋转的各种表示
  LOG(INFO) << "----------- Eigen and GTSAM Rotation -----------";
  // 注意旋转的归一化，保证创建的矩阵是旋转矩阵，或者四元数是单位四元数
  // 要么构建前对旋转向量归一化 Eigen::Vector3d().normalize，要么对构建后的旋转量归一化 Eigen::Quaterniond().normalize()
  Eigen::Quaterniond tmp_R = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d(1, 2, 3)));  //.normalized()
  tmp_R.normalize();
  LOG(INFO) << "!!! Eigen !!!";
  LOG(INFO) << "R: " << tmp_R.toRotationMatrix();
  LOG(INFO) << "q(xyzw): " << tmp_R.coeffs().transpose();
  auto vec = Eigen::AngleAxisd(tmp_R);
  LOG(INFO) << "AngleAxisd: " << vec.angle() * vec.axis().transpose();
  LOG(INFO) << "angle: " << vec.angle() / M_PI * 180;
  LOG(INFO) << "axis: " << vec.axis().transpose();

  LOG(INFO) << "!!! log_Quaternion !!!";
  auto vec_2 = log_Quaternion(tmp_R);
  LOG(INFO) << "AngleAxisd: " << vec_2.transpose();
  LOG(INFO) << "!!! GTSAM !!!";
  auto R_gtsam = gtsam::Rot3(tmp_R);
  LOG(INFO) << "R: " << R_gtsam.matrix();
  LOG(INFO) << "q(wxyz): " << R_gtsam.quaternion().transpose();
  LOG(INFO) << "angle: " << R_gtsam.axisAngle().second / M_PI * 180;

  gtsam::Vector axis_gtsam(R_gtsam.axisAngle().first.point3());
  LOG(INFO) << "axis: " << axis_gtsam.transpose();
  LOG(INFO) << "AngleAxisd: " << (axis_gtsam * R_gtsam.axisAngle().second).transpose();
  LOG(INFO) << "Logmap: " << gtsam::Rot3::Logmap(R_gtsam).transpose();

  return 0;
}