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
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../estimator/feature_manager.h"

using namespace Eigen;
using namespace std;
// 存储一帧图像的全部信息
class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
		// 该帧图像的特征点信息
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
		// 从帧body坐标系到参考帧相机坐标系(后面会变为世界坐标系)的变化：w_R_b
        Matrix3d R;
		// 从帧相机坐标系到参考相机坐标系(后面会变为世界坐标系)的位移：w_t_c
        Vector3d T;
		// 从上一帧到该帧图像的IMU预积分结果
        IntegrationBase *pre_integration;
		// 该帧是否为关键帧
        bool is_key_frame;
};
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs);
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);