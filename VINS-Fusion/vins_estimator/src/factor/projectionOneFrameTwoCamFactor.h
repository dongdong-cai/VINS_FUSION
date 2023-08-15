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

#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../estimator/parameters.h"

class ProjectionOneFrameTwoCamFactor : public ceres::SizedCostFunction<2, 7, 7, 1, 1>
{
  public:
    ProjectionOneFrameTwoCamFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
    				   			   const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
    	   			   			   const double _td_i, const double _td_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);
	// 特征点在i帧和j帧的归一化相机坐标系坐标
    Eigen::Vector3d pts_i, pts_j;
	// 特征点在i帧和j帧的移动速度
    Eigen::Vector3d velocity_i, velocity_j;
	// 第i帧和第j帧的IMU和cam的时间误差
    double td_i, td_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
