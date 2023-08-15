/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}
/**
 * @brief 把读取到的配置文件参数设置到位姿估计器中
 */
void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
	// sqrt_info是信息矩阵
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
	// 如果是多线程，就会启动多线程运行processMeasurements
	// 如果是多线程模式，就是一个线程做光流，一个线程做后端优化，否则，就是一个做完光流之后在做线程优化,串行处理
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}
/**
 * @brief 输入一帧左右目图片(或者单目)，会执行VO优化得到位姿
 * @param t 图片时间戳
 * @param _img 左目图片
 * @param _img1 右目图片，如果为单目相机则为空矩阵
 */
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
	// 1.提取特征点
    if(_img1.empty()) // 单目
        featureFrame = featureTracker.trackImage(t, _img);
    else  //双目
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

	// 2.发布加上特征点后的图片（用蓝点和红点标注跟踪次数不同的特征点，用绿点标注上一帧识别到的特征点）
    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

	// 3.开启processMeasurements函数线程，后端非线性优化IMU和图片数据，得到位姿
    if(MULTIPLE_THREAD)  // 多线程
    {
		// 进行降采样，两帧图片只处理一帧，因为当相机采用频率高的时候会出现计算延迟现象
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
			// 将时间戳和图片特征点存入featureBuf中
            featureBuf.push(make_pair(t, featureFrame));
			// 这里没有调用函数processMeasurements是因为多线程在前面estimator.setParameter()中已经开启了该函数的线程
            mBuf.unlock();
        }
    }
    else // 单线程
    {
        mBuf.lock();
		// 将时间戳和图片特征点存入featureBuf中
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
		// 后端非线性优化图片和IMU数据，得到优化后的位姿
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}
/**
 * @brief 输入imu数据供后端的非线性优化，每当订阅到新的imu话题数据就会执行该函数，当认为imu初始化完成后会调用fastPredictIMU
 * 函数更新位置P、旋转Q、速度V，并发布到话题imu_propagate中，话题更新的频率和imu频率相同
 *
 * @param t 时间戳
 * @param linearAcceleration 线加速度
 * @param angularVelocity 角速度
 */
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
		// 快速对IMU预积分得到当前位姿
        fastPredictIMU(t, linearAcceleration, angularVelocity);
		// 将当前位姿结果发布出去
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}

/**
 * @brief 在IMU buf中提取两帧图像之间的IMU数据
 * @param t0 上一帧图像的时间戳
 * @param t1 当前帧图像的时间戳
 * @param accVector [out] 两帧图像之间的加速度数据
 * @param gyrVector [out] 两帧图像之间的角速度数据
 * @return
 */
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
		// 把IMU buf中上一帧时间前的数据全部pop
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
		// 把两帧图像时间之间的IMU buf保存到accVector、gyrVector中
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
		// 最后再存入一帧时间大于t1的IMU 数据
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}
/**
 * @brief 判断时间t有无超出IMU buf中现有的最迟数据的时间，也就是粗判断IMU buf中有无该时间戳的数据
 * @param t 时间戳
 * @return
 */
bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}
/**
 * @brief 对IMU数据和图像特征点数据非线性优化处理，得到相机位姿：
 * 1.IMU积分更新位姿
 * 2.利用当前帧图像信息，进行后续的关键帧判断、初始化、非线性优化、边缘化、滑动窗口移动等操作
 * 3.将结果用ROS话题发布出去
 */
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
		// 存储一帧帧图像的特征点信息，key是时间戳，value是特征点数据值
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
		// 两帧图像之间的IMU数据，key是时间戳，value是IMU数据值
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if(!featureBuf.empty())
        {
			// 1.取出当前帧识别的特征点信息
            feature = featureBuf.front();
			// td：是时间补偿，秦通博士处理：将td认为是一个常值（在极短时间内是不变化的）
			// 由于触发器等各种原因，IMU和图像帧之间存在时间延迟，因此需要进行补偿
			// 详见Online_Temporal_Calibration_for_Monocular_Visual-Inertial_Systems
            curTime = feature.first + td;
			// 2.等待合适的IMU数据
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
					// 有时候程序一直报wait for imu ...，可能是因为配置文件中的td设置太大了
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
			// 3.把前一帧和当前帧图片之间的IMU信号提取出来，存入accVector、gyrVector中
            if(USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            mBuf.unlock();
			// 4.IMU积分更新位姿
            if(USE_IMU)
            {
				// 初始化第一帧的位姿，因为IMU不是水平放置，所以Z轴和{0, 0, 1.0}对齐，通过对齐获得Rs[0]的初始位姿
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
					// 计算两帧IMU数据之间的dt
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
					// IMU积分更新位姿
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();
			// 5.利用当前帧图像信息，进行后续的关键帧判断、初始化、非线性优化、边缘化、滑动窗口移动等操作
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);
			// 6.发布ROS话题
            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }
		// 这里如果不允许多线程就会退出这个循环，也就是每次Estimator::inputImage输入图片，然后执行该函数
		// 结束该函数，然后等待下一次Estimator::inputImage，成为串行的结构
        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}
/**
 * @brief 因为IMU不一定是水平放置，那么IMU的Z轴和{0, 0, 1.0}不对齐
 *        通过对齐，得到 g 到 {0, 0, 1.0} 的旋转（消除绕Z轴的旋转），作为初始位姿Rs[0]
 * @param accVector 加速度和g在短时间是紧耦合的，但是短时间内g占accVector的大部分值，可以通过accVector求得g的估计值
 * @return
 */
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
	// 计算第一帧图像时间内IMU加速度数据的平均值
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
	// 将IMU的Z轴与重力方向的对齐转换初值 w_R_g
    Matrix3d R0 = Utility::g2R(averAcc);
	// 下面两行，感觉重复了，已经在Utility::g2R函数里面执行过了
    // double yaw = Utility::R2ypr(R0).x();
    // R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/**
 * @brief IMU预积分，积分结果会作为后端非线性优化的初始值，包括RPV和delta_RPV
 * @param t 当前IMU数据的时间戳
 * @param dt 当前帧和上一帧的相隔时间
 * @param linear_acceleration 当前帧的线加速度
 * @param angular_velocity 当前帧的角速度
 */
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
	// 如果是该函数第一次处理IMU数据，就把当前IMU数据保存为上一帧的IMU数据
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
	// 如果这帧图像到上一帧图像还没有创建滑动窗口的IMU预积分类，就创建之
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
	// 如果不是第一帧还需要计算IMU积分结果
    if (frame_count != 0)
    {
		// 1.计算得到的IMU预积分pre_integrations[frame_count]（这里预积分结果是相对值PVQ，即P代表该时刻相对于上一帧图像的位置变化），作为后端非线性优化的一个输入
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
			// 这里没必要重复计算
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
		// 2.利用前一帧图像和当前帧图像之间的IMU信号，更新当前帧的位姿Rs[i]、Ps[i]、Vs[i]，作为后端非线性优化的初始值
		// Rs Ps Vs是frame_count这一个图像帧开始的预积分值,是在绝对坐标系下的.（这里预积分结果是当前的绝对值PVQ，即P就代表该时刻的位置）
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}
/**
 * @brief 对图像特征点和IMU预积分结果进行后端非线性优化：
 * 1.判断当前帧是否为关键帧，同时完成特征点和帧之间关系的建立
 * 2.进行camera到IMU(body)外参的标定
 * 3.如果还没有初始化，先完成初始化
 * 4.如果完成了初始化，就进行后端优化
 * @param image 当前帧左右目图像特征点 feature_id：(camera_id：(x, y, z, p_u, p_v, velocity_x, velocity_y))
 * @param header 时间戳
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
	// VINS为了减少优化的计算量，只优化滑动窗口内的帧，因此保证滑动窗口内帧的质量很关键，每来新的一帧是一定会加入到滑动窗口中的，但是
	// 挤出去上一帧还是窗口最旧帧是依据新的一帧是否为关键帧决定，保证了滑动窗口中处理最新帧可能不是关键帧，其他帧都会是关键帧

	// 1.判断当前帧是否为关键帧，同时完成特征点和帧之间关系的建立
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
		// 如果为关键帧，则边缘化滑动窗口中最旧的帧
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
		// 如果不是关键帧，则边缘化滑动窗口的上一帧（为什么不是关键帧要把次新帧踢出去，因为是否关键帧是判断其与前面几帧的视差大不）
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
	// 把输入图像插入到all_image_frame中
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
	// 2、进行camera到IMU(body)外参的标定
    if(ESTIMATE_EXTRINSIC == 2) // ESTIMATE_EXTRINSIC表示不提供配置文件里面的初始值
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
			// 找到frame_count - 1帧和frame_count帧的匹配特征点对
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
			// 在线标定一个imu_T_cam外参作为初始值
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
				// 不初始化右目的外参？
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
				// 然后就可以变为提供初始值的外参标定记号了
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
	// 3.如果还没有初始化，先完成初始化
	// 初始化感觉传感器类型可分为三种模式：单目+IMU、双目+IMU、双目
    if (solver_flag == INITIAL)
    {
        // 3.1  单目+IMU初始化
        if (!STEREO && USE_IMU)
        {
			// 单目+IMU模式需要等到滑动窗口满了才可以，可能是因为相比较双目，单目的尺度不确定性，需要多帧及特征点恢复
			// frame_count此时还没有更新，所以当前帧的ID是frame_count+1，也就是说现在一共有WINDOW_SIZE+1帧
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
				// 距离上一次初始化时间间隔超过0.1s才会进行新的初始化
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
					// 初始化成功了，就执行后端非线性优化
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // 3.2  双目+IMU初始化
        if(STEREO && USE_IMU)
        {
			// PnP求解当前帧的位姿：w_T_imu
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
			// 恢复所有帧的特征点深度
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
			// 如果滑动窗口第一次满了
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
				// 陀螺仪偏置校正，并根据更新后的bg进行IMU积分更新
                solveGyroscopeBias(all_image_frame, Bgs);
				// 依据新的IMU的加速度和角速度偏置值，重新IMU预积分
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
				// 初始化成功了，就执行后端非线性优化
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // 3.3  双目初始化
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();
			// 如果滑动窗口满了，就执行后端的非线性优化
            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }
		// 如果滑动窗口还没有满就添加到滑动窗口中
        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
			// 注意，这里frame_count已经是下一帧的索引了，这里就是把当前帧估计都位姿当做下一帧的初始值
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
	// 4.如果完成了初始化，就进行后端优化
	// 在完成初始化后就只进行后端非线性优化了，还是需要将滑窗中的特征点尽可能多地恢复出对应的3D点，获取多帧之间更多的约束，
	// 进而得到更多的优化观测量, 使得优化结果更加鲁棒
    else
    {
        TicToc t_solve;

		// PnP求解当前帧位姿
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		// 三角化特征点
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
		// 执行非线性优化
        optimization();
		// 将重投影误差过大的点删除
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
		// 如果不是多线程（因为多线程会降采样），那么可以从前端中的pre_cts中移除外点，避免再次跟踪；并预测预测下一帧上特征点的像素坐标
        if (! MULTIPLE_THREAD)
        {
			// 通知前端feature tracker移除这些
            featureTracker.removeOutliers(removeIndex);
			// 通过匀速模型预测下一帧上特征点的像素坐标
            predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());
		// 系统故障检测
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
		// 滑窗更新
        slideWindow();
		// 将长期跟踪但是深度仍未成功恢复的点删除
        f_manager.removeFailures();
        // prepare output of VINS
		// 更新一些状态变量
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }  
}
/**
 * @brief 单目+IMU模式的初始化
 * @return bool，初始化是否成功
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // 1.计算滑动窗口内IMU加速度的标准差，用于判断移动快慢
    {
        map<double, ImageFrame>::iterator frame_it;
		// 滑动窗口内全部帧的平均加速度总和
        Vector3d sum_g;
		// 遍历当前滑动窗口内全部帧，统计其IMU加速度平均和，注意for里面没有算上all_image_frame的第一帧
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
			// 该帧的IMU积分总时长
            double dt = frame_it->second.pre_integration->sum_dt;
			// IMU积分的速度变化量 / 总时间 = 平均加速度
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
		// 滑动窗口内帧与帧之间的平均加速度，由于sum_g没有算上第一帧，所以要减1
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
			// 计算加速度的方差
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
		// 计算加速度的标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
		// 如果滑动窗口内的IMU加速度标准差小于0.25，会认为初始化时移动不够剧烈，可能导致数据误差较大，不利用初始化
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
			// 这里作者把return false注释了，所以这里计算加速度标准差没用到
            //return false;
        }
    }
    // 2.在滑动窗口中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变化
	// 如果相对运动恢复失败，结束本次初始化；如果相对运动恢复成功, 继续进行SFM初始化工作

	// 存储每一帧相机坐标系到第l帧相机坐标系（参考帧相机坐标系）的相对位姿
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
	// 存储求解出来的在第l帧相机坐标系下的全部特征点的3D坐标
    map<int, Vector3d> sfm_tracked_points;
	// 滑动窗口内识别到的全部特征点队列
    vector<SFMFeature> sfm_f;
	// 2.1 遍历滑动窗口内的全部特征点,都保存为SFMFeature数据类型
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
		// 把该特征点出现过的图像ID和对应图像上的2D位置都存进去
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
	// 找到的历史关键帧和当前帧的相对运动：历史帧_R_当前帧
    Matrix3d relative_R;
    Vector3d relative_T;
	// 找到的历史参考关键帧的ID
    int l;
	// 2.2 在滑动窗口中找到满足以下条件的历史关键帧，计算该历史关键帧和当前帧的相对位姿变化：
	// 1）匹配特征点对超过20
	// 2）视差足够大，且在求解Rt过程中内点对数大于12
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
	// 3.SFM优化：以上面找到的l历史关键帧为参考系（该帧会作为系统原点），Pnp计算滑动窗口每帧位姿，然后三角化所有特征点，构建BA优化每帧位姿
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
		// 如果SFM中BA失败，则初始化失败，并且边缘化最旧帧
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // 4.SFM只恢复了滑动窗口内关键帧的位姿，现在把滑动窗口外的帧位姿也恢复
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
	// 遍历目前为止的全部图像
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
		// 如果遍历到滑动窗口里面的关键帧
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
			// RIC：表示从相机系到body系的转换
			// 从该帧body坐标系到参考帧相机坐标系的变化
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
		// 不是滑动窗口里面的帧，就同样的方法恢复出位姿
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
		// 遍历该帧图像中的所有特征点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
			// 遍历观测到该特征点的所有观测值，并使用pts_3_vector、pts_2_vector记录
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
		// 根据pts_3_vector、pts_2_vector中记录的3D、2D点信息，恢复相对位姿
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
	// 5.视觉-惯性对齐（对相机和IMU采用松耦合的方式）
	// 前面我们用摄像头可以计算相邻两帧之间的相机旋转，用IMU可以积分得到相邻两帧之间的积分旋转，现在我们需要调整二者的参数使二者的旋转结果重合
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}
/**
 * @brief 视觉-惯性对齐：用相邻两帧之间的相机旋转和IMU积分旋转，构建最小二乘问题，优化陀螺仪的零偏b、速度、重力和相机的尺度等参数：
 * 前面我们用摄像头可以计算相邻两帧之间的相机旋转，用IMU可以积分得到相邻两帧之间的积分旋转，现在我们需要调整二者的参数使二者的旋转结果重合
 * @return
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
	// 需要优化的变量：各帧的速度、重力、尺度
    VectorXd x;
    // 1.通过视觉和IMU对齐，完成陀螺仪偏置的校正并初始化当前所有帧的速度以及重力和尺度因子
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // 2.根据更新的Bgs、Velocity、Gravity、Scale重新计算当前滑窗中位姿
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }
	// 取出尺度初始值
    double s = (x.tail<1>())(0);
	// 用更新后的陀螺仪零偏，更新滑动窗口中各帧的IMU预积分结果
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
		// 这里作者认为初始化时加速度计的零偏影响不大，所以直接设置为0
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
	// 尺度恢复，将平移矫正到参考帧相机坐标系上
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
	// 速度较正到参考帧相机坐标系上
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
	// 通过重力加速度的方向，计算从参考帧坐标系转到Z轴朝上的世界坐标系的旋转矩阵
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
	// 并将所有量都恢复至世界坐标系中（也就是说定义的世界坐标系就是将参考帧坐标系的Z轴微调到重力方向笔直朝上的坐标系）
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());
	// ATTENTION：将SFM恢复的特征点深度重置，用修复后的pose重新进行三角化
    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}
/**
 * @brief 在历史滑动窗口中寻找适合于当前帧求解相对位姿的关键帧，该历史关键帧需要同时满足两个条件：
 * 1.匹配特征点对超过20
 * 2.视差足够大，且在求解Rt过程中内点对数大于12
 * @param relative_R 求解出的历史关键帧与当前关键帧的R，历史帧_R_当前帧
 * @param relative_T 求解出的历史关键帧与当前关键帧的T，历史帧_t_当前帧
 * @param l 该历史关键帧的ID（历史关键帧是从最旧帧开始寻找，找到了一个满足条件的就退出），会作为参考关键帧
 * @return 是否有找到该历史关键帧
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
	// 遍历滑动窗口中的每一帧
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
		// 得到滑动窗口的历史帧与当前帧的匹配特征点对
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
		// 条件一：匹配特征点对超过20
        if (corres.size() > 20)
        {
			// 计算匹配特征点对的平均像素距离（视差）
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
			// 条件二：视差足够大，且在求解Rt过程中内点对数大于12
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}
/**
 * @brief 将所有待优化变量转变为double数组形式，因为ceres不能直接优化向量vector
 */
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}
/**
 * @brief 将所有待优化变量从ceres求解结果的double数组形式转变为eigen类型
 */
void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}
/**
 * @brief 后端非线性优化的主体函数，待优化的状态变量共包括滑动窗口内的n+1个所有相机的状态
 * （包括位置、朝向、速度、加速度计bias和陀螺仪bias）、Camera到IMU的外参、m+1个3D点的逆深度
 */
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
	// 1、将所有待优化变量转变为double数组形式，因为ceres不能直接优化向量vector
    vector2double();

	// 2、构建ceres优化问题problem，和损失函数loss_function
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0); // 核函数
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

	// 3、添加待优化变量，即向ceres问题中添加参数块
    for (int i = 0; i < frame_count + 1; i++)
    {
		// 由于姿态不满足正常的加法，也就是李群上没有加法，因此需要自己定义它的加法
		// 具体是在PoseLocalParameterization中的子类local_parameterization中的plus里面实现的
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
		// 3.1  添加滑窗中位姿 world_t_imu(维度3)、world_R_imu(维度4)
		// problem.AddParameterBlock是添加待优化的状态变量
		// 参数1：所添加的参数块  参数2：所添加的参数块大小  参数3：参数块中自增更新的方式，如果不提供就默认按照ceres自带的更新方法
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
			// 3.2  添加速度和加速度计以及陀螺仪的偏置 w_V_imu(维度3)、Bas(维度3)、Bgs(维度3)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
		// 不用IMU，这样就是六自由度不可观了，所以索性fix第一帧
        problem.SetParameterBlockConstant(para_Pose[0]);

	// 3.3 添加外参 imu_t_cam(维度3)、imu_R_cam(维度4)
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
		/*
            第一次优化外参时，需要满足下列条件：
            ESTIMATE_EXTRINSIC：是否需要估计外参
            frame_count == WINDOW_SIZE：仅当滑窗内帧数达到最大时，才优化外参
            Vs[0].norm() > 0.2：还是需要一些运动激励

            之后，openExEstimation = 1 是不改变的，所以会一直优化外参
        */
		if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
			// 如果不优化外参，就将外参固定
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
	// 3.4  添加时间补偿，认为IMU和camera之间存在一个时间差异，需要进行时间同步td：cam_time + td = imu_time
    problem.AddParameterBlock(para_Td[0], 1);

	// 如果没有时间同步的需求或者运动激励不够，则将时间同步td固定
    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

	// 4.开始添加ceres残差块，即相应约束
	// 4.1  边缘化带来的先验约束
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // 用上一次的边缘化结果作为这一次的先验约束
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
		// problem.AddResidualBlock是添加约束
		// 参数1: costFunction    参数2: 核函数(NULL)    参数3: costFunction函数需要的形参
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
	// 4.2  IMU预积分约束
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
			// 时间过长这个预积分约束就不可信了  若预积分的累积积分的时间跨度超过10s，则不能形成约束
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
			// 预积分类IMUFactor中实现的是残差，残差对待优化变量的雅可比以及对应的协方差矩阵
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }
	// 4.3  视觉约束
	// 视觉约束数目
    int f_m_cnt = 0;
	// 遍历时当前特征点的索引
    int feature_index = -1;
	// 遍历每一个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
		// 仅添加性能好的特征点带来的视觉约束
        if (it_per_id.used_num < 4)
            continue;
		// TODO：这个是不是应该放在continue之前？？？
		// 放在continue之后，就有问题：如果某个特征点被跳过了，但是没有计数，那么索引就会出错
        ++feature_index;
		// 第一个观测到这个特征点的帧idx  逆深度是与这一帧所绑定的
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        // 该特征点在发现的第一帧归一化相机坐标系下的坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
		// 遍历看到这个特征点的所有KF，把全部可以共同看到该特征点的两帧形成一个视觉约束
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
			// 两个相同帧不能形成重投影
            if (imu_i != imu_j)
            {
				// 取出另一帧的归一化相机坐标
                Vector3d pts_j = it_per_frame.point;
				// 同一个摄像头的不同时刻两帧观测到同一个特征点形成的约束
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
				// 如果是双目，那么和右目那一帧也可以组成共视的两帧约束，也就是两个摄像头的不同时刻两帧观测到同一个特征点
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
				// 而且即使两帧是同一帧，此时左右目也可以组成共视的两帧约束，不过此时无法对位姿形成约束，但是可以对外参和特征点深度形成约束
                else
                {
					// 两个摄像头在同一时刻的两帧观测到同一个特征点
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

	// 5.ceres求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR; // ceres求解类型(稠密求解)
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;// dogleg求解
    options.max_num_iterations = NUM_ITERATIONS;// 优化求解的最大迭代次数
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;

	// 下面的边缘化老帧操作比较多，因此给他优化时间就少一些
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

	// 6、将优化后的变量从double数组转换为vector类型，后续继续使用
    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;

	// 7.边缘化操作
	// 个人理解就是滑动窗口中帧与帧、帧与IMU之间存在许多约束，当我们将滑动窗口中某一帧去掉后，和该帧有关的约束就全不能用了，有些浪费，边缘化操作
	// 就是想办法把这些和该帧有关的约束变为其他帧之间的约束，相当于该帧虽然被剔除了，但一些约束依旧留了下来，比如说三个人a,b,c
	// a和b相距5m，a和c相距3m，当我们把a剔除后，这两个约束就不能用了，我们可以在剔除a前面把这两个约束变为b和c相距8m，最大程度保留约束信息
	// 简单粗暴的理解就是人给我滚，财产给我留下

    TicToc t_whole_marginalization;
	// 7.1 前面若认为最新帧是关键帧，则将最老的一帧相关信息边缘化掉
    if (marginalization_flag == MARGIN_OLD)
    {
		// 实例化一个管理边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
		// 因此后面用到的costFunction是和double数组绑定的，故也需要将eigen状态量都转成double数组
        vector2double();
		/*
		  关于边缘化有几点注意的地方
		  1、找到需要边缘化的参数块，这里是地图点，第0帧位姿，第0帧速度零偏
		  2、找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，具体如下:
			 a. 预积分残差: 待边缘化的第0帧与第1帧之间的预积分残差，所关联到的状态量有两帧之间的位姿和速度零偏
			 b. 视觉重投影残差约束: 待边缘化的第0帧上的特征点与其他帧的特征点之间能够投影到同一地图点，形成视觉重投影约束. 它所关联到的状态量有: 第0帧以及与之共视的滑窗中的其他帧，3d地图点的逆深度，相机与IMU之间的外参.
			 c. 上一次边缘化约束
		  3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏
		*/

		// 上一次边缘化约束的边缘化
        if (last_marginalization_info && last_marginalization_info->valid)
        {
			// 需要被边缘化的状态变量
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
				// 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
				// 因为本次要边缘化最老帧，我们把上一次边缘化留下的约束中和最老帧位姿和速度有关的约束都扔了
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
			// 参数1：costFunction
			// 参数2：核函数(NULL)
			// 参数3：costFunction的状态变量
			// 参数4：在参数3的状态变量中，有哪些变量需要被边缘化，这里是第0帧的位姿、速度、bias
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
			// 往边缘化对象中加入残差块
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
		// 第1帧与第0帧之间存在预积分约束的边缘化
        if(USE_IMU)
        {
			// 若预积分的累积积分的时间跨度超过10，则不能形成约束，时间太长不可信
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
				// 这里参数4表示预积分约束中需要被边缘化的变量是参数3[0]、参数3[1]，也就是第0帧的位姿、速度、bias
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
		// 视觉重投影约束的边缘化
        {
            int feature_index = -1;
			// 遍历全部特征点
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
				// 只选择质量高的特征点
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
				// 只要与imu_i(第0帧)形成视觉重投影约束
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
				// 遍历看到这个特征点的所有KF，通过这个特征点，建立和第0帧的约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
						// 同一个特征点被一个相机在不同时刻两帧观测到的重投影约束
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        // 边缘化的变量是第0帧的位姿以及3D地图点(因为边缘化会将稀疏矩阵变成稠密矩阵，为避免矩阵维度过大，将3D地图点均边缘化掉)
						ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
							// 同一个特征点被两个相机在不同时刻两帧观测到的重投影约束
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
							// 边缘化的变量是第0帧的位姿以及3D地图点(因为边缘化会将稀疏矩阵变成稠密矩阵，为避免矩阵维度过大，将3D地图点均边缘化掉)
							ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
							// 同一个特征点被两个相机在同一时刻两帧观测到的重投影约束
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
							// 边缘化的变量是3D地图点(因为边缘化会将稀疏矩阵变成稠密矩阵，为避免矩阵维度过大，将3D地图点均边缘化掉)
							ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
		// 进行预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
		// 边缘化操作  主要是计算出新的增量方程的雅可比和残差
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());
		// 即将滑窗，因此记录新地址对应的老地址
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
			// 位姿和速度都要滑窗移动    老地址 = 新地址
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
		// 外参和时间延时不变(不需要在滑窗中维护)
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
		// parameter_blocks实际上就是addr_shift的索引的集合及搬进去的新地址
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
		// 清空上一次边缘化信息
        if (last_marginalization_info)
            delete last_marginalization_info;
		// 将当前边缘化信息存放到上一次边缘化信息变量中
        last_marginalization_info = marginalization_info;
		// 代表该次边缘化对某些参数块形成约束，这些参数块在滑窗之后的地址
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
	// 7.2  若最新帧不是关键帧，则将次新帧边缘化掉
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
			// 边缘化最老帧会对三个残差进行边缘化，但这里只对前一帧边缘化约束进行边缘化，不对IMU预积分和视觉重投影约束进行边缘化

			// 定义边缘化管理器
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
			// 在上一次边缘化约束，找到其中本次被边缘化的变量
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
				// 遍历上一次边缘化的变量
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
					// 找到本次边缘化的变量：次新帧的位姿、速度、bias
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // 上一次边缘化约束的边缘化
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
			// 预处理
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
			// 边缘化
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
			// 滑窗准备
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            // 更新变量
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}
/**
 * @brief 滑动窗口处理
 */
void Estimator::slideWindow()
{
    TicToc t_margin;
	// 根据边缘化种类的不同，进行滑窗的方式也不同
	// 1.边缘化最老帧
    if (marginalization_flag == MARGIN_OLD)
    {
		// 边缘化之前，将最老帧的位姿进行备份
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
		// 必须是填满了滑窗才可以
        if (frame_count == WINDOW_SIZE)
        {
			// 一帧一帧交换过去，以此将最老帧(第0帧)移到滑窗的最后位置上(具体是指位姿和速度零偏)
			// 假设WINDOW_SIZE = 5
			// 0 1 2 3 4 5
			// 1 0 2 3 4 5
			// 1 2 0 3 4 5
			// 1 2 3 0 4 5
			// 1 2 3 4 0 5
			// 1 2 3 4 5 0
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
			// 最后一帧的状态量赋上当前值(也是此时的最新值)，作为后续状态传递的初始值
			// 1 2 3 4 5 0  ==>  1 2 3 4 5 5
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
				// 将最后一帧的IMU预积分量清零
                delete pre_integrations[WINDOW_SIZE];
				// 利用当前的状态量计算新的最后一帧的预积分
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
				// buffer清空，等待新的IMU数据来填
                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
			// 清空all_image_frame最老帧之前的状态(时间戳t_0之前的)
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
			// 地图点的交接问题处理
            slideWindowOld();
        }
    }
    else
		// 2.边缘化次新帧
    {
        if (frame_count == WINDOW_SIZE)
        {
			// 0 1 2 3 4 5 => 0 1 2 3 5 5
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}
/**
 * @brief 对被移除的倒数第二帧的地图点进行处理
 */
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
/**
 * @brief 由于地图点是绑定在第一个看见它的位姿上的，因此需要对被移除的帧看见的地图点进行解绑，以及每个地图点的首个观测帧id减1
 */
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
	// 如果初始化过了  表示3D地图点已经被正确地恢复出来了
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
		// back_R0、back_P0是前面备份的被移除的帧的位姿(在IMU坐标系下)
		// R0 P0是当前被移除的滑动窗口中最老的相机的姿态
		// R1 P1是当前滑动窗口移除掉最老帧后新的最老帧 （这两帧一个是原最老，一个是原第二老）
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
		// 把被移除帧(R0和P0)看见地图点的管理权交给当前的最老帧(R1和P1)
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
		// 如果初始化不成功
        f_manager.removeBack();
}

/**
 * @brief 获取当前帧的世界坐标系位姿T
 * @param T
 */
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}
/**
 * @brief 获取index帧的世界坐标系位姿T
 * @param index
 * @param T
 */
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}
/**
 * @brief 通过匀速模型预测下一帧能够跟踪到的特征点的像素坐标
 */
void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
	// 分别获得当前帧和上一帧的位姿
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
	// 使用匀速模型估计下一帧的位姿
    nextT = curT * (prevT.inverse() * curT);
	// 特征点id->预测帧相机坐标系坐标
    map<int, Eigen::Vector3d> predictPts;
	// 遍历所有的特征点
    for (auto &it_per_id : f_manager.feature)
    {
		// 只预测有有效深度的特征点
        if(it_per_id.estimated_depth > 0)
        {
			// 计算观测到该点的首帧和末帧的索引
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
			// 如果他的末帧等于当前滑窗的最后一帧，说明没有跟丢，才有预测下一帧位置的可能
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
				// 特征点位置转到imu坐标系
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
				// 转到世界坐标系
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
				// 转到预测的下一帧的imu坐标系下去
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
				// 转到下一帧的相机坐标系
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}
/**
 * @brief 计算重投影误差
 * @return
 */
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}
/**
 * @brief 将重投影误差过大的点视为外点
 * @param removeIndex[in,out] 外点的索引
 */
void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
	// 遍历全部特征点
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
		// 该特征点被观测的次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();
		// 观测帧数小于4就先不看
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
		// imu_i是第一次观测到该特征点的帧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
		// 遍历全部观测到该特征点的关键帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
			// 同一个特征点被一个相机在不同时刻观测到
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
				// 计算重投影误差
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
				// 同一个特征点被两个相机在不同时刻观测到
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
				// 同一个特征点被两个相机在同一时刻观测到
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
		// 该特征点的平均重投影误差
        double ave_err = err / errCnt;
		// 归一化相机 * 焦距 = 像素，如果误差像素超过3就认为是outlier
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}
/**
 * @brief IMU预积分，快速积分当前IMU数据得到相机位姿，但误差较大
 * @param t 当前时间戳
 * @param linear_acceleration 当前线加速度
 * @param angular_velocity 当前角速度
 */
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
	// 前一帧加速度（世界系）
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
	// 更新旋转Q （eq 15-34）
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
	// 更新加速度
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // 这里加速度取两个时刻的平均值
	// 更新位置P（eq 15-43）
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
	// 更新速度V（eq 15-40）
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}
/**
 * @brief 更新一些变量
 */
void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
