/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"

/**
 * @brief 用于自定义状态变量是如何实现增量的，即 x_plus_delta = x + delta
 * @param x 维度7，world_t_imu(维度3)、world_R_imu(维度4)
 * @param delta 状态增量
 * @param x_plus_delta 状态加上增量后的结果
 * @return
 */
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
	// 将前3维double类型的数据x(表示位移)映射成eigen的Vector3d类型
    Eigen::Map<const Eigen::Vector3d> _p(x);
	// x中的旋转姿态是以四元数的形式表示的, 因此将double类型中x的后4维映射成四元数类型_q
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
	// 由于增量delta的前3维表示位移, 将前3个数放入Vector3d类型的dp中，则dp表示位移增量
    Eigen::Map<const Eigen::Vector3d> dp(delta);
	// 以四元数形式表示的姿态增量
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));
	// 存放位移相加结果
    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
	// 存放四元数姿态相加结果
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);
	// 位移直接相加即可
    p = _p + dp;
	// 四元数的增量，库已经重载了*号，但是需要归一化，因为只有模为1的单位四元数才能表示旋转
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
