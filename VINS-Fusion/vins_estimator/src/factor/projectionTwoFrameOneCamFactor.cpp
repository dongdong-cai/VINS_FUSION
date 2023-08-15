/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "projectionTwoFrameOneCamFactor.h"

Eigen::Matrix2d ProjectionTwoFrameOneCamFactor::sqrt_info;
double ProjectionTwoFrameOneCamFactor::sum_t;

ProjectionTwoFrameOneCamFactor::ProjectionTwoFrameOneCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j, 
                                       const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                       const double _td_i, const double _td_j) : 
                                       pts_i(_pts_i), pts_j(_pts_j), 
                                       td_i(_td_i), td_j(_td_j)
{
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;

#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};
/**
 * @brief 计算视觉重投影残差及其雅可比矩阵
 * @param parameters 待优化的状态变量
 * @param residuals 视觉重投影的残差，计算残差是为了知道啥时候迭代结束
 * @param jacobians 计算的雅可比矩阵，是为了知道优化状态变量的增量方向
 * @return
 */
bool ProjectionTwoFrameOneCamFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
	// 视觉重投影中待优化的第i帧位姿 world_t_imu(维度3)、world_R_imu(维度4)
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
	// 视觉重投影中待优化的第j帧位姿 world_t_imu(维度3)、world_R_imu(维度4)
    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
	// 视觉重投影中待优化的相机与IMU之间的外参 imu_t_cam(维度3)、imu_R_cam(维度4)
    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
	// 视觉重投影中待优化的逆深度 1/depth
    double inv_dep_i = parameters[3][0];
	// 相机和imu话题的时间差 cam_time + td = imu_time
    double td = parameters[4][0];
	// 根据话题时间差微调特征点在归一化相机坐标系坐标
    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;
	// 计算第i帧特征点的相机坐标系坐标
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
	// 计算第i帧特征点的IMU坐标系坐标
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
	// 计算第i帧特征点的世界坐标系坐标
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
	// 计算第i帧特征点在第j帧的imu坐标系坐标
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
	// 计算第i帧特征点在第j帧的相机坐标系坐标
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
	// 重投影残差就是第i帧特征点在第j帧归一化相机坐标系的坐标和第j帧特征点在归一化相机坐标系的坐标的误差
	// 先把pts_camera_j转换到第j帧归一化相机坐标系，然后计算残差
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
	// 乘以信息矩阵
    residual = sqrt_info * residual;

    if (jacobians)
    {
		// 第i帧位姿 world_R_imu
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
		// 第j帧位姿 world_R_imu
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
		// 外参 imu_R_cam
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3); // 链式求导的第一步
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
		// 先求链式求导的第一步：残差reduce对第j帧重投影相机坐标系坐标p_{c_j}的雅可比矩阵
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce; // 误差乘上信息矩阵

        if (jacobians[0])
        {
			// p_{c_j}对第i帧位姿的导数
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
			// 对位置的求导
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
			// 对姿态的求导
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
			// p_{c_j}对第j帧位姿的导数
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
			// 对位置的求导
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
			// 对姿态的求导
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
			// p_{c_j}对外参的导数
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
			// 对位置的求导
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
			// 对姿态的求导
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
			// p_{c_j}对逆深度的导数
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
        }
        if (jacobians[4])
        {
			// p_{c_j}对td的导数
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
            jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                          sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}

void ProjectionTwoFrameOneCamFactor::check(double **parameters)
{
    double *res = new double[2];
    double **jaco = new double *[5];
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 1];
    jaco[4] = new double[2 * 1];
    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Vector2d>(jaco[4]) << std::endl
              << std::endl;

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    double inv_dep_i = parameters[3][0];
    double td = parameters[4][0];

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i ) * velocity_i;
    pts_j_td = pts_j - (td - td_j ) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Vector2d residual;

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 20> num_jacobian;
    for (int k = 0; k < 20; k++)
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        double inv_dep_i = parameters[3][0];
        double td = parameters[4][0];


        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            Pi += delta;
        else if (a == 1)
            Qi = Qi * Utility::deltaQ(delta);
        else if (a == 2)
            Pj += delta;
        else if (a == 3)
            Qj = Qj * Utility::deltaQ(delta);
        else if (a == 4)
            tic += delta;
        else if (a == 5)
            qic = qic * Utility::deltaQ(delta);
        else if (a == 6 && b == 0)
            inv_dep_i += delta.x();
        else if (a == 6 && b == 1)
            td += delta.y();

        Eigen::Vector3d pts_i_td, pts_j_td;
        pts_i_td = pts_i - (td - td_i) * velocity_i;
        pts_j_td = pts_j - (td - td_j) * velocity_j;
        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        Eigen::Vector2d tmp_residual;

#ifdef UNIT_SPHERE_ERROR 
        tmp_residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
        tmp_residual = sqrt_info * tmp_residual;

        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian.block<2, 6>(0, 0) << std::endl;
    std::cout << num_jacobian.block<2, 6>(0, 6) << std::endl;
    std::cout << num_jacobian.block<2, 6>(0, 12) << std::endl;
    std::cout << num_jacobian.block<2, 1>(0, 18) << std::endl;
    std::cout << num_jacobian.block<2, 1>(0, 19) << std::endl;
}
