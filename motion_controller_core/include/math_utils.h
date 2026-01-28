#ifndef MATH_TYPE_DEFINE_H
#define MATH_TYPE_DEFINE_H


#define DEG2RAD (0.01745329251994329576923690768489)
// constexpr size_t MAX_DOF=50;
#define COD_THRESHOLD 1.0E-6

#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#define GRAVITY 9.80665
#define MAX_DOF 50U
#define RAD2DEG 1/DEG2RAD


namespace Eigen
{

	// Eigen default type definition
	#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
	typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
	typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;  \
	typedef Matrix<Type, 1, Size>    RowVector##SizeSuffix##TypeSuffix;

	typedef double	rScalar;

	EIGEN_MAKE_TYPEDEFS(rScalar, d, 5, 5)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 6, 6)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 7, 7)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 8, 8)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 12, 12)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 18, 18)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 28, 28)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 30, 30)
	EIGEN_MAKE_TYPEDEFS(rScalar, d, 32, 32)

	// typedef Transform<rScalar, 3, Eigen::Isometry> HTransform;  // typedef Transform< double, 3, Isometry > 	Eigen::Isometry3d

	typedef Matrix<rScalar, 1, 3>	Matrix1x3d;
	typedef Matrix<rScalar, 1, 4>	Matrix1x4d;
	typedef Matrix<rScalar, 4, 3>	Matrix4x3d;
	typedef Matrix<rScalar, 6, 3>	Matrix6x3d;
	typedef Matrix<rScalar, 6, 7>	Matrix6x7d;
	typedef Matrix<rScalar, 8, 4>	Matrix8x4d;
	typedef Matrix<rScalar, -1, 1, 0, MAX_DOF, 1> VectorJXd;
	typedef Matrix<rScalar, -1, 1, 0, 12, 1> VectorLXd; //Leg IK
	typedef Matrix<rScalar, -1, -1, 0, MAX_DOF, MAX_DOF> MatrixJXd;

	//Complex
	typedef Matrix<std::complex<double>, 8, 4> Matrix8x4cd;

}

namespace MathUtils
{

	//constexpr double GRAVITY {9.80665};
	//constexpr double DEG2RAD {};

	static double cubic(double time,     ///< Current time
		double time_0,   ///< Start time
		double time_f,   ///< End time
		double x_0,      ///< Start state
		double x_f,      ///< End state
		double x_dot_0,  ///< Start state dot
		double x_dot_f   ///< End state dot
	)
	{
		double x_t;

		if (time < time_0)
		{
			x_t = x_0;
		}
		else if (time > time_f)
		{
			x_t = x_f;
		}
		else
		{
			double elapsed_time = time - time_0;
			double total_time = time_f - time_0;
			double total_time2 = total_time * total_time;  // pow(t,2)
			double total_time3 = total_time2 * total_time; // pow(t,3)
			double total_x = x_f - x_0;

			x_t = x_0 + x_dot_0 * elapsed_time

				+ (3 * total_x / total_time2
					- 2 * x_dot_0 / total_time
					- x_dot_f / total_time)
				* elapsed_time * elapsed_time

				+ (-2 * total_x / total_time3 +
				(x_dot_0 + x_dot_f) / total_time2)
				* elapsed_time * elapsed_time * elapsed_time;
		}

		return x_t;
	}

	static double cubicDot(double time,     ///< Current time
		double time_0,   ///< Start time
		double time_f,   ///< End time
		double x_0,      ///< Start state
		double x_f,      ///< End state
		double x_dot_0,  ///< Start state dot
		double x_dot_f   ///< End state dot
	)
	{
		double x_t;

		if (time < time_0)
		{
			x_t = x_dot_0;
		}
		else if (time > time_f)
		{
			x_t = x_dot_f;
		}
		else
		{
			double elapsed_time = time - time_0;
			double total_time = time_f - time_0;
			double total_time2 = total_time * total_time;  // pow(t,2)
			double total_time3 = total_time2 * total_time; // pow(t,3)
			double total_x = x_f - x_0;

			x_t = x_dot_0

				+ 2 * (3 * total_x / total_time2
					- 2 * x_dot_0 / total_time
					- x_dot_f / total_time)
				* elapsed_time

				+ 3 * (-2 * total_x / total_time3 +
				(x_dot_0 + x_dot_f) / total_time2)
				* elapsed_time * elapsed_time;
		}

		return x_t;
	}

	static Eigen::Matrix3d skew(Eigen::Vector3d src)
	{
		Eigen::Matrix3d skew;
		skew.setZero();
		skew(0, 1) = -src[2];
		skew(0, 2) = src[1];
		skew(1, 0) = src[2];
		skew(1, 2) = -src[0];
		skew(2, 0) = -src[1];
		skew(2, 1) = src[0];

		return skew;
	}

	template <int N>
	static Eigen::Matrix<double, N, 1> cubicVector(double time,     ///< Current time
		double time_0,   ///< Start time
		double time_f,   ///< End time
		Eigen::Matrix<double, N, 1> x_0,      ///< Start state
		Eigen::Matrix<double, N, 1> x_f,      ///< End state
		Eigen::Matrix<double, N, 1> x_dot_0,  ///< Start state dot
		Eigen::Matrix<double, N, 1> x_dot_f   ///< End state dot
	)
	{

		Eigen::Matrix<double, N, 1> res;
		for (unsigned int i = 0; i<N; i++)
		{
			res(i) = cubic(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
		}
		return res;
	}

	static Eigen::VectorXd cubicVector(double time,
									   double time_0,
									   double time_f,
									   const Eigen::VectorXd& x_0,
									   const Eigen::VectorXd& x_f,
									   const Eigen::VectorXd& x_dot_0,
									   const Eigen::VectorXd& x_dot_f)
	{
		assert(x_0.size() == x_f.size());
		Eigen::VectorXd res(x_0.size());
		for (int i = 0; i < x_0.size(); ++i)
		{
			res(i) = cubic(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
		}
		return res;
	}

	template <int N>
	static Eigen::Matrix<double, N, 1> cubicDotVector(double time,     ///< Current time
		double time_0,   ///< Start time
		double time_f,   ///< End time
		Eigen::Matrix<double, N, 1> x_0,      ///< Start state
		Eigen::Matrix<double, N, 1> x_f,      ///< End state
		Eigen::Matrix<double, N, 1> x_dot_0,  ///< Start state dot
		Eigen::Matrix<double, N, 1> x_dot_f   ///< End state dot
	)
	{

		Eigen::Matrix<double, N, 1> res;
		for (unsigned int i = 0; i<N; i++)
		{
			res(i) = cubicDot(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
		}
		return res;
	}

	static Eigen::VectorXd cubicDotVector(double time,
                                   double time_0,
                                   double time_f,
                                   const Eigen::VectorXd& x_0,
                                   const Eigen::VectorXd& x_f,
                                   const Eigen::VectorXd& x_dot_0,
                                   const Eigen::VectorXd& x_dot_f)
	{
		assert(x_0.size() == x_f.size());
		Eigen::VectorXd res(x_0.size());
		for (int i = 0; i < x_0.size(); ++i)
		{
			res(i) = cubicDot(time, time_0, time_f, x_0(i), x_f(i), x_dot_0(i), x_dot_f(i));
		}
		return res;
	}

	// Original Paper
	// Kang, I. G., and F. C. Park.
	// "Cubic spline algorithms for orientation interpolation."
	const static Eigen::Matrix3d rotationCubic(double time,
		double time_0,
		double time_f,
		const Eigen::Matrix3d &rotation_0,
		const Eigen::Matrix3d &rotation_f)
	{
		if (time >= time_f)
		{
			return rotation_f;
		}
		else if (time < time_0)
		{
			return rotation_0;
		}
		double tau = cubic(time, time_0, time_f, 0, 1, 0, 0);
		Eigen::Matrix3d rot_scaler_skew;
		rot_scaler_skew = (rotation_0.transpose() * rotation_f).log();
		Eigen::Matrix3d result = rotation_0 * (rot_scaler_skew * tau).exp();

		return result;
	}

	const static Eigen::Vector3d rotationCubicDot(
		double time, double time_0, double time_f,
		const Eigen::Vector3d &w_0, const Eigen::Vector3d &a_0,
		const Eigen::Matrix3d &rotation_0, const Eigen::Matrix3d &rotation_f)
	{
		Eigen::Matrix3d r_skew;
		r_skew = (rotation_0.transpose() * rotation_f).log();
		Eigen::Vector3d a, b, c, r;
		double tau = (time - time_0) / (time_f - time_0);
		r(0) = r_skew(2, 1);
		r(1) = r_skew(0, 2);
		r(2) = r_skew(1, 0);
		c = w_0;
		b = a_0 / 2;
		a = r - b - c;
		Eigen::Vector3d rd;
		for (int i = 0; i < 3; i++)
		{
			rd(i) = cubicDot(time, time_0, time_f, 0, r(i), 0, 0);
		}
		rd = rotation_0 * rd;
		if (tau < 0) return w_0;
		if (tau > 1) return Eigen::Vector3d::Zero();
		return rd;//3 * a * pow(tau, 2) + 2 * b * tau + c;
	}
	
	static Eigen::Vector3d getPhi(Eigen::Matrix3d current_rotation,
		Eigen::Matrix3d desired_rotation)
	{
		Eigen::Vector3d phi;
		Eigen::Vector3d s[3], v[3], w[3];

		for (int i = 0; i < 3; i++) {
			v[i] = current_rotation.col(i);
			w[i] = desired_rotation.col(i);
			s[i] = v[i].cross(w[i]);
		}
		phi = s[0] + s[1] + s[2];
		phi = -0.5* phi;

		return phi;
	}

	static Eigen::Isometry3d multiplyIsometry3d(Eigen::Isometry3d A,
		Eigen::Isometry3d B)
	{
		Eigen::Isometry3d AB;

		AB.linear() = A.linear()*B.linear();
		AB.translation() = A.linear()*B.translation() + A.translation();
		return AB;
	}

	static Eigen::Isometry3d inverseIsometry3d(Eigen::Isometry3d A)
	{
		Eigen::Isometry3d A_inv;

		A_inv.linear() = A.linear().transpose();
		A_inv.translation() = -A.linear().transpose()*A.translation();
		return A_inv;
	}

	static Eigen::Matrix3d rotateWithZ(double yaw_angle)
	{
		Eigen::Matrix3d rotate_wth_z(3, 3);

		rotate_wth_z(0, 0) = cos(yaw_angle);
		rotate_wth_z(1, 0) = sin(yaw_angle);
		rotate_wth_z(2, 0) = 0.0;

		rotate_wth_z(0, 1) = -sin(yaw_angle);
		rotate_wth_z(1, 1) = cos(yaw_angle);
		rotate_wth_z(2, 1) = 0.0;

		rotate_wth_z(0, 2) = 0.0;
		rotate_wth_z(1, 2) = 0.0;
		rotate_wth_z(2, 2) = 1.0;

		return rotate_wth_z;
	}

	static Eigen::Matrix3d rotateWithY(double pitch_angle)
	{
		Eigen::Matrix3d rotate_wth_y(3, 3);

		rotate_wth_y(0, 0) = cos(pitch_angle);
		rotate_wth_y(1, 0) = 0.0;
		rotate_wth_y(2, 0) = -sin(pitch_angle);

		rotate_wth_y(0, 1) = 0.0;
		rotate_wth_y(1, 1) = 1.0;
		rotate_wth_y(2, 1) = 0.0;

		rotate_wth_y(0, 2) = sin(pitch_angle);
		rotate_wth_y(1, 2) = 0.0;
		rotate_wth_y(2, 2) = cos(pitch_angle);

		return rotate_wth_y;
	}

	static Eigen::Matrix3d rotateWithX(double roll_angle)
	{
		Eigen::Matrix3d rotate_wth_x(3, 3);

		rotate_wth_x(0, 0) = 1.0;
		rotate_wth_x(1, 0) = 0.0;
		rotate_wth_x(2, 0) = 0.0;

		rotate_wth_x(0, 1) = 0.0;
		rotate_wth_x(1, 1) = cos(roll_angle);
		rotate_wth_x(2, 1) = sin(roll_angle);

		rotate_wth_x(0, 2) = 0.0;
		rotate_wth_x(1, 2) = -sin(roll_angle);
		rotate_wth_x(2, 2) = cos(roll_angle);

		return rotate_wth_x;
	}

	static Eigen::Vector3d rot2Euler(Eigen::Matrix3d Rot)
	{
		double beta;
		Eigen::Vector3d angle;
		beta = -asin(Rot(2, 0));

		if (abs(beta) < 90 * DEG2RAD)
			beta = beta;
		else
			beta = 180 * DEG2RAD - beta;

		angle(0) = atan2(Rot(2, 1), Rot(2, 2) + 1E-37); //roll
		angle(2) = atan2(Rot(1, 0), Rot(0, 0) + 1E-37); //pitch
		angle(1) = beta; //yaw

		return angle;
	}

	template <typename _Matrix_Type_>
	static _Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
	{
		Eigen::JacobiSVD< _Matrix_Type_ > svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
		double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);

		return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
	}


	static void floatGyroframe(Eigen::Isometry3d trunk, Eigen::Isometry3d reference, Eigen::Isometry3d new_trunk)
	{
		Eigen::Vector3d rpy_ang;
		rpy_ang = MathUtils::rot2Euler(reference.linear());

		Eigen::Matrix3d temp;
		temp = MathUtils::rotateWithZ(-rpy_ang(2));

		new_trunk.linear() = temp * trunk.linear();
		new_trunk.translation() = temp * (trunk.translation() - reference.translation());
	}


	template <int _State_Size_, int _Input_Size_>
	Eigen::Matrix<double, _State_Size_, _State_Size_> discreteRiccatiEquation(
		Eigen::Matrix<double, _State_Size_, _State_Size_> a,
		Eigen::Matrix<double, _State_Size_, _Input_Size_> b,
		Eigen::Matrix<double, _Input_Size_, _Input_Size_> r,
		Eigen::Matrix<double, _State_Size_, _State_Size_> q)
	{
		Eigen::Matrix4d z11, z12, z21, z22;
		z11 = a.inverse();
		z12 = a.inverse()*b*r.inverse()*b.transpose();
		z21 = q * a.inverse();
		z22 = a.transpose() + q * a.inverse()*b*r.inverse()*b.transpose();

		Eigen::Matrix<double, 2 * _State_Size_, 2 * _State_Size_> z;
		z.setZero();
		z.topLeftCorner(4, 4) = z11;
		z.topRightCorner(4, 4) = z12;
		z.bottomLeftCorner(4, 4) = z21;
		z.bottomRightCorner(4, 4) = z22;

		std::vector<double> eigVal_real(8);
		std::vector<double> eigVal_img(8);
		std::vector<Eigen::Vector8d> eigVec_real(8);
		std::vector<Eigen::Vector8d> eigVec_img(8);

		for (int i = 0; i<8; i++)
		{
			eigVec_real[i].setZero();
			eigVec_img[i].setZero();
		}

		Eigen::Matrix<double, 2 * _State_Size_, 1> deigVal_real, deigVal_img;
		Eigen::Matrix<double, 2 * _State_Size_, 2 * _State_Size_> deigVec_real, deigVec_img;
		deigVal_real.setZero();
		deigVal_img.setZero();
		deigVec_real.setZero();
		deigVec_img.setZero();
		deigVal_real = z.eigenvalues().real();
		deigVal_img = z.eigenvalues().imag();

		Eigen::EigenSolver<Eigen::Matrix<double, 2 * _State_Size_, 2 * _State_Size_>> ev(z);
		//EigenVector Solver
		//Matrix3D ones = Matrix3D::Ones(3,3);
		//EigenSolver<Matrix3D> ev(ones);
		//cout << "The first eigenvector of the 3x3 matrix of ones is:" << endl << ev.eigenvectors().col(1) << endl;

		for (int i = 0; i<8; i++)
		{
			for (int j = 0; j<8; j++)
			{
				deigVec_real(j, i) = ev.eigenvectors().col(i)(j).real();
				deigVec_img(j, i) = ev.eigenvectors().col(i)(j).imag();
			}
		}

		//Order the eigenvectors
		//move e-vectors correspnding to e-value outside the unite circle to the left

		Eigen::Matrix8x4d tempZ_real, tempZ_img;
		tempZ_real.setZero();
		tempZ_img.setZero();
		int c = 0;

		for (int i = 0; i<8; i++)
		{
			if ((deigVal_real(i)*deigVal_real(i) + deigVal_img(i)*deigVal_img(i))>1.0) //outside the unit cycle
			{
				for (int j = 0; j<8; j++)
				{
					tempZ_real(j, c) = deigVec_real(j, i);
					tempZ_img(j, c) = deigVec_img(j, i);
				}
				c++;
			}
		}

		Eigen::Matrix8x4cd tempZ_comp;
		for (int i = 0; i<8; i++)
		{
			for (int j = 0; j<4; j++)
			{
				tempZ_comp.real()(i, j) = tempZ_real(i, j);
				tempZ_comp.imag()(i, j) = tempZ_img(i, j);
			}
		}

		Eigen::Matrix4cd U11, U21, X;
		for (int i = 0; i<4; i++)
		{
			for (int j = 0; j<4; j++)
			{
				U11(i, j) = tempZ_comp(i, j);
				U21(i, j) = tempZ_comp(i + 4, j);
			}
		}
		X = U21 * (U11.inverse());
		Eigen::Matrix4d X_sol;
		for (int i = 0; i<4; i++)
		{
			for (int j = 0; j<4; j++)
			{
				X_sol(i, j) = X.real()(i, j);
			}
		}

		return X_sol;
	}

	static Eigen::Vector3d legGetPhi(Eigen::Isometry3d rotation_matrix1, Eigen::Isometry3d active_r1, Eigen::Vector6d ctrl_pos_ori)
	{
		Eigen::Matrix3d active_r, rotation_matrix, x_rot, y_rot, z_rot, d_rot, s1_skew, s2_skew, s3_skew;
		x_rot.setZero();
		y_rot.setZero();
		z_rot.setZero();
		d_rot.setZero();
		s1_skew.setZero();
		s2_skew.setZero();
		s3_skew.setZero();

		active_r = active_r1.linear();

		x_rot = rotateWithX(ctrl_pos_ori(3));
		y_rot = rotateWithY(ctrl_pos_ori(4));
		z_rot = rotateWithZ(ctrl_pos_ori(5));
		d_rot = active_r.inverse()*z_rot*y_rot*x_rot;

		rotation_matrix = active_r.inverse() * rotation_matrix1.linear();

		s1_skew = skew(rotation_matrix.col(0));
		s2_skew = skew(rotation_matrix.col(1));
		s3_skew = skew(rotation_matrix.col(2));

		Eigen::Vector3d s1f, s2f, s3f, phi;
		s1f.setZero();
		s2f.setZero();
		s3f.setZero();

		s1f = s1_skew * d_rot.col(0);
		s2f = s2_skew * d_rot.col(1);
		s3f = s3_skew * d_rot.col(2);

		phi = (s1f + s2f + s3f) * (-1.0 / 2.0);

		return phi;
	}

	static Eigen::MatrixXd PinvCOD(const Eigen::MatrixXd &A)
    {
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A.rows(), A.cols());
        cod.setThreshold(COD_THRESHOLD);
        cod.compute(A);

        return cod.pseudoInverse();
    }

	// template <typename _Matrix_Type_>
    // static void PinvCOD(const _Matrix_Type_ &A, _Matrix_Type_ &ret)
    // {
    //     Eigen::CompleteOrthogonalDecomposition<_Matrix_Type_> cod(A.rows(), A.cols());
    //     cod.setThreshold(COD_THRESHOLD);
    //     cod.compute(A);

    //     ret = cod.pseudoInverse();
    // }

	// template <typename _Matrix_Type_>
    // static void PinvCOD(const _Matrix_Type_ &A, _Matrix_Type_ &ret, _Matrix_Type_ &V2)
    // {
    //     int rows = A.rows();
    //     int cols = A.cols();
    //     Eigen::CompleteOrthogonalDecomposition<_Matrix_Type_> cod(rows, cols);
    //     cod.setThreshold(COD_THRESHOLD);
    //     cod.compute(A);
    //     int rank = cod.rank();
    //     _Matrix_Type_ Vtemp = cod.householderQ().transpose();
    //     V2 = (Vtemp).block(rank, 0, rows - rank, cols);

    //     ret = cod.pseudoInverse();
    // }

	inline double getRBF(const double &h, const double &delta=-0.5)
	{
		double result;
		if (h >= delta) result = log(h+1);
		else            result = log(delta+1) + 1/(delta+1) * (h-delta) - 1/(2*pow(delta+1,2)) * pow(h-delta,2);
		return result;
	}

	inline Eigen::VectorXd getRBF(const Eigen::VectorXd &h, const double &delta=-0.5)
	{
		Eigen::VectorXd result(h.size());
		for(size_t i=0; i<result.size(); i++) result(i) = getRBF(delta, h(i));
		return result;
	}

	inline double getDRBF(const double &h, const double &delta=-0.5)
	{
		double result;
		if (h >= delta) result = 1/(h+1);
		else            result = 1/(delta+1) - 1/(pow(delta+1,2)) * (h-delta);
		return result;
	}

	inline Eigen::VectorXd getDRBF(const Eigen::VectorXd &h, const double &delta=-0.5)
	{
		Eigen::VectorXd result(h.size());
		for(size_t i=0; i<result.size(); i++) result(i) = getDRBF(delta, h(i));
		return result;
	}

    static void getTaskSpaceError(const Eigen::Affine3d& x_target, 
						          const Eigen::Vector6d& xdot_target,
						          const Eigen::Affine3d& x,
						          const Eigen::Vector6d& xdot,
						          Eigen::Vector6d& x_error,
						          Eigen::Vector6d& xdot_error)
	{
		x_error.setZero();
		xdot_error.setZero();
		x_error.head(3) = x_target.translation() - x.translation();
		x_error.tail(3) = getPhi(x_target.rotation(), x.rotation());
		xdot_error = xdot_target - xdot;
	}

    static void getTaskSpaceCubic(const Eigen::Affine3d& x_target,
						          const Eigen::Vector6d& xdot_target,
						          const Eigen::Affine3d& x_init,
						          const Eigen::VectorXd& xdot_init,
						          const double& current_time,
						          const double& init_time,
						          const double& duration,
						          Eigen::Affine3d& x_desired,
						          Eigen::Vector6d& xdot_desired)
	{
		x_desired.setIdentity();
		xdot_desired.setZero();
		x_desired.translation() = cubicVector(current_time,
											  init_time,
											  init_time + duration,
											  x_init.translation(),
											  x_target.translation(),
											  xdot_init.head(3),
											  xdot_target.head(3));
		x_desired.linear() = rotationCubic(current_time,
										   init_time,
										   init_time + duration,
										   x_init.rotation(),
										   x_target.rotation());
		xdot_desired.head(3) = cubicDotVector(current_time,
											  init_time,
											  init_time + duration,
											  x_init.translation(),
											  x_target.translation(),
											  xdot_init.head(3),
											  xdot_target.head(3));
		xdot_desired.tail(3) = rotationCubicDot(current_time,
												init_time,
												init_time + duration,
												Eigen::Vector3d::Zero(),
												Eigen::Vector3d::Zero(),
												x_init.rotation(),
												x_target.rotation());
	}

}
#endif