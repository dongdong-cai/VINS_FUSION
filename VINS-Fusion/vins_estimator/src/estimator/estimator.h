/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"


class Estimator
{
  public:
    Estimator();
    ~Estimator();
    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
	// 更新数据时的线程锁
    std::mutex mBuf;
    std::mutex mPropagate;
	// imu的加速度数据buf，key是时间，value是加速度
    queue<pair<double, Eigen::Vector3d>> accBuf;
	// imu的角速度数据buf，key是时间，value是角速度
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
	// 特征点Buf，key是时间戳，value是左右目特征点信息（具体数据结构看FeatureTracker::trackImage）
	// 特征点信息内容为：feature_id：(camera_id：(x, y, z, p_u, p_v, velocity_x, velocity_y))
	// feature_id为特征点ID 、camera_id为相机ID（0为左目，1为右目）、(x, y, z)为特征点在归一化相机坐标系的3D坐标
	// (p_u, p_v)为特征点在图像坐标系的2D坐标、(velocity_x, velocity_y)为特征点在图像坐标系的速度
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
	// 图像上一帧时间，图像当前帧时间
    double prevTime, curTime;
	// 是否优化外参 imu_T_cam
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;
	// 特征点跟踪器
    FeatureTracker featureTracker;
	// 求解器状态：初始化|非线性优化
    SolverFlag solver_flag;
	// 边缘化策略 MARGIN_OLD | MARGIN_SECOND_NEW
    MarginalizationFlag  marginalization_flag;
	// 重力
    Vector3d g;
	// 相机的外参矩阵imu_R_cam，最多两个相机
    Matrix3d ric[2];
	// 相机的外参矩阵imu_t_cam，最多两个相机
    Vector3d tic[2];
	// 滑动窗口，待优化的位置，world_t_imu，刚开始是通过IMU预积分得到初始值，后面会通过相机进一步优化
    Vector3d        Ps[(WINDOW_SIZE + 1)];
	// 滑动窗口，待优化的速度，w_V_imu
    Vector3d        Vs[(WINDOW_SIZE + 1)];
	// 滑动窗口，待优化的位姿：world_R_imu
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
	// 滑动窗口，待优化的加速度计零偏
    Vector3d        Bas[(WINDOW_SIZE + 1)];
	// 滑动窗口，待优化的陀螺仪零偏
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
	// 相机和imu话题的时间差：cam_time + td = imu_time
    double td;
	// back_R0、back_P0是暂时保存的待边缘化的最老关键帧的位姿:world_T_imu
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
	// 滑动窗口，记录窗口内每一帧图像的时间戳
    double Headers[(WINDOW_SIZE + 1)];
	// 滑动窗口，保存每两帧图像之间的IMU预积分类
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
	// 上一帧的imu数据
    Vector3d acc_0, gyr_0;
	// 滑动窗口，记录每个窗口中每两帧imu之间的dt
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
	// 滑动窗口，记录每个窗口内的imu加速度大小
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
	// 滑动窗口，记录每个窗口内的imu角速度大小
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
	// 滑动窗口内，当前图像是第几帧（索引），当滑动窗口满了后，frame_count不会再继续++了，一直保持WINDOW_SIZE的大小
    int frame_count;
	// sum_of_back是边缘化最老帧的次数、sum_of_front是边缘化次新帧的次数
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
	// 输入图片数目的计数器
    int inputImageCnt;
	// 特征点管理器
    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;
	// 是否已经处理过第一帧的imu数据
    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
	// 上一次进行初始化的时间
    double initial_timestamp;

	// ceres中待优化的状态变量：
	// 待优化的滑动窗口状态：world_t_imu(维度3)、world_R_imu(维度4)
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
	// 待优化的滑动窗口状态：w_V_imu(维度3)、Bas(维度3)、Bgs(维度3)
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
	// 待优化的滑动窗口状态：特征点的逆深度(维度1)
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
	// 待优化的滑动窗口状态：imu_t_cam(维度3)、imu_R_cam(维度4)
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
	// 待优化的滑动窗口状态：相机和imu话题的时间差 cam_time + td = imu_time
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;
	// 上一次边缘化信息变量
    MarginalizationInfo *last_marginalization_info;
	// 上一次边缘化对哪些当前参数块有约束
    vector<double *> last_marginalization_parameter_blocks;
	// 包含全部帧的ImageFrame key是时间戳，value是ImageFrame（包含了该帧全部特征点信息，和IMU预积分结果）
	// all_image_frame在滑动窗口处理函数slideWindow中也会逐渐清理掉边缘化了的关键帧
    map<double, ImageFrame> all_image_frame;
	// 从上一帧图像到当前帧图像的IMU预积分结果
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;
	// 上一时刻的时间
    double latest_time;
	// 上一时刻的位置P、速度V、加速度零偏Ba、陀螺仪零偏Bg、加速度acc_0、角速度gyr_0
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
	// 上一时刻的位姿Q_w_b
    Eigen::Quaterniond latest_Q;
	// 是否完成了第一帧的位姿初始化
    bool initFirstPoseFlag;
    bool initThreadFlag;
};
