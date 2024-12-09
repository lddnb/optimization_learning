#include <random>
#include <execution>
#include <numeric>

#include <glog/logging.h>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

class CeresCostFunctor 
{
  public:
    CeresCostFunctor(const Eigen::Vector3d& curr_point, const Eigen::Vector3d& target_point)
    : curr_point_(curr_point), target_point_(target_point) {}

    template <typename T>
    bool operator()(const T* const se3, T* residual) const {
        Eigen::Map<const Eigen::Quaternion<T>> R(se3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(se3 + 4);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_eigen(residual);
        
        residual_eigen = R * curr_point_.cast<T>() + t - target_point_.cast<T>();
        return true;
    }

  private:
    Eigen::Vector3d curr_point_;
    Eigen::Vector3d target_point_;
};

class GtsamIcpFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GtsamIcpFactor(gtsam::Key key, const gtsam::Point3& source_point, const gtsam::Point3& target_point, const gtsam::SharedNoiseModel& cost_model)
            : gtsam::NoiseModelFactor1<gtsam::Pose3>(cost_model, key), source_point_(source_point), target_point_(target_point) {}

	virtual gtsam::Vector evaluateError(const gtsam::Pose3& T, boost::optional<gtsam::Matrix&> H = boost::none) const override
	{
		gtsam::Matrix A = gtsam::Matrix::Zero(3, 6);

		gtsam::Point3 p_trans = T.transformFrom(source_point_, A);
		gtsam::Vector error = p_trans - target_point_;
		if (H) {
			*H = A;
		}
		return error;
	}

private:
	gtsam::Point3 source_point_;
  gtsam::Point3 target_point_;
};

int main(int argc, char **argv){

    std::vector<Eigen::Vector3d> source_points;
    std::vector<Eigen::Vector3d> target_points;
    
    Eigen::Quaterniond R_ture = Eigen::Quaterniond(Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d t_ture = Eigen::Vector3d(1, 2, 3);

    // Eigen 中四元数乘以向量和转成旋转矩阵后再乘以向量的结果是一样的
    // 可以都看成是旋转向量的运算
    // 下面的三种方式等价，第三种才是用四元数乘法的形式实现的，即 q' = q * [0, v] * q.inverse()
    // LOG(INFO) << "1: " <<  R_ture * t_ture;
    // LOG(INFO) << "2: " <<  R_ture.toRotationMatrix() * t_ture;
    // LOG(INFO) << "3: " <<  (R_ture * Eigen::Quaterniond(0, t_ture.x(), t_ture.y(), t_ture.z()) * R_ture.inverse()).vec();

    const int num_points = 10; // 生成100个点
    const double min_value = -100.0; // 最小值
    const double max_value = 100.0;  // 最大值

    // 创建随机数生成器
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(min_value, max_value);

    // 生成随机点
    for (int i = 0; i < num_points; ++i) {
        double x = distribution(generator); // x
        double y = distribution(generator); // y
        double z = distribution(generator); // z
        source_points.emplace_back(Eigen::Vector3d(x, y, z));
        target_points.emplace_back(R_ture * source_points.back() + t_ture);
    }

    // initial guess
    Eigen::Quaterniond R_init = Eigen::Quaterniond(Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d t_init = Eigen::Vector3d(2, 3, 4);

    // 一、ceres 优化求解方法
    double T[7] = {R_init.x(), R_init.y(), R_init.z(), R_init.w(), t_init.x(), t_init.y(), t_init.z()};

    ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>> *se3 = 
        new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
    ceres::Problem problem;
    problem.AddParameterBlock(T, 7, se3);

    // AutoDiffCostFunction 函数模板参数为
    // CeresCostFunctor 类，3 为 residual size，7 为 parameter size，其实就是操作符函数中各个指针的长度
    // 如果有多个输入参数，则依次在后面加上各个参数的指针长度
    for(int i = 0; i < num_points; i++){
        ceres::CostFunction *cost_function = 
            new ceres::AutoDiffCostFunction<CeresCostFunctor, 3, 7>(
                new CeresCostFunctor(source_points[i], target_points[i]));
        problem.AddResidualBlock(cost_function, nullptr, T);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << summary.FullReport();

    // 这里的 Eigen::Map 相当于是用的 Eigen::Quaterniond q(Eigen::Vector4d(x, y, z, w)) 的形式初始化
    // 这才是在几个数据在内存中的布局，而不是 Eigen::Quaterniond q(w, x, y, z) 的形式
    // 下面的三种方式等价
    // 1. Eigen::Map<Eigen::Quaterniond> R(T);
    // 2. Eigen::Quaterniond R{T[3], T[0], T[1], T[2]};
    // 3. Eigen::Quaterniond R(Eigen::Vector4d(T[0], T[1], T[2], T[3]));
    Eigen::Map<Eigen::Quaterniond> R(T);
    Eigen::Map<Eigen::Vector3d> t(T + 4);

    LOG(INFO) << "R_init: " << R_init.coeffs().transpose();
    LOG(INFO) << "t_init: " << t_init.transpose();
    LOG(INFO) << "R_Ture: " << R_ture.coeffs().transpose();
    LOG(INFO) << "t_ture: " << t_ture.transpose();
    LOG(INFO) << "----------- Ceres -----------";
    LOG(INFO) << "R: " << R.coeffs().transpose();
    LOG(INFO) << "t: " << t.transpose();

    // 二、 SVD 分解求法
		LOG(INFO) << "----------- SVD -----------";
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d source_bar = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_bar = Eigen::Vector3d::Zero();
    
    source_bar = std::reduce(std::execution::par, source_points.begin(), source_points.end(), source_bar) / source_points.size();
    target_bar = std::reduce(std::execution::par, target_points.begin(), target_points.end(), target_bar) / target_points.size();
    // source_bar = std::accumulate(source_points.begin(), source_points.end(), source_bar) / source_points.size();
    // target_bar = std::accumulate(target_points.begin(), target_points.end(), target_bar) / target_points.size();

    // LOG(INFO) << "source_bar: " << source_bar.transpose();
    // LOG(INFO) << "target_bar: " << target_bar.transpose();

    for(int i = 0; i < num_points; i++){
        A += (source_points[i] - source_bar) * (target_points[i] - target_bar).transpose();
    }
    LOG(INFO) << "det(A): " << A.determinant();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
    Eigen::Vector3d t_svd = target_bar - R_svd * source_bar;

		LOG(INFO) << "det(R): " << R_svd.determinant();
    LOG(INFO) << "R: " << Eigen::Quaterniond(R_svd).coeffs().transpose();
    LOG(INFO) << "t: " << t_svd.transpose();

    // 三、 gtsam 优化求解方法
		LOG(INFO) << "----------- GTSAM -----------";
    const gtsam::Pose3 T_true = gtsam::Pose3(gtsam::Rot3(R_ture), gtsam::Point3(t_ture));
    std::vector<gtsam::Point3> source_gtsam(source_points.begin(), source_points.end());
    std::vector<gtsam::Point3> target_gtsam(target_points.begin(), target_points.end());
    // gtsam::noiseModel::Isotropic cost_model = gtsam::noiseModel::Isotropic(3, 1.0);
    gtsam::SharedNoiseModel noise_model = gtsam::noiseModel::Isotropic::Sigma(3 , 1);
    const gtsam::Key key = gtsam::symbol_shorthand::X(0);
    gtsam::NonlinearFactorGraph graph;
		for (int i = 0; i < num_points; i++) {
    	graph.emplace_shared<GtsamIcpFactor>(key, source_gtsam[i], target_gtsam[i], noise_model);
		}
    gtsam::Values initial_estimate;
		gtsam::Pose3 T_init = gtsam::Pose3(gtsam::Rot3(R_init), gtsam::Point3(t_init));
    initial_estimate.insert(key, T_init);
		gtsam::GaussNewtonParams params_gn;
		params_gn.maxIterations = 100;
		params_gn.relativeErrorTol = 1e-5;
    gtsam::GaussNewtonOptimizer optimizer(graph, initial_estimate);
    optimizer.optimize();
    gtsam::Values result = optimizer.values();
    gtsam::Pose3 T_result = result.at<gtsam::Pose3>(key);

    LOG(INFO) << "R: " << Eigen::Quaterniond(T_result.rotation().matrix()).coeffs().transpose();
    LOG(INFO) << "t: " << T_result.translation().transpose();

    return 0;
}

