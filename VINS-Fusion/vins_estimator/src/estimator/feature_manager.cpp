/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}
/**
 * @brief 设置ric矩阵
 * @param _ric ric矩阵的设置值
 */
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}
/**
 * @brief 统计feature容器中跟踪次数超过4次的特征点数目
 * @return
 */
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief 把特征点放入feature的list容器中，即建立特征点和观测帧之间的联系，同时判断当前帧是否是关键帧，这决定了是边缘化滑动窗口中的次新帧还是最旧帧：
 * 1.该帧新的特征点很多，认为是关键帧
 * 2.该帧新特征点虽然不多，但是如果特征点在上一帧和上上帧中视差变化大，则也认为是关键帧，否则不是关键帧
 * @param frame_count 滑动窗口内，当前图像是第几帧（索引）
 * @param image 当前帧的特征点信息 feature_id：(camera_id：(x, y, z, p_u, p_v, velocity_x, velocity_y))
 * @param td IMU和cam的时间误差 cam_time + td = imu_time
 * @return bool true：是关键帧，边缘化最旧帧;false：不是关键帧，边缘化次新帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
	// 所有特征点视差总和
    double parallax_sum = 0;
	// 满足某些条件的特征点数目
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
	// 1.遍历当前帧的全部特征点，如果是第一次观测到的特征点，就加入到feature大家庭中；如果是已出现的特征点，只需更新观测次数即可
    for (auto &id_pts : image)
    {
		// 添加左目的特征点
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
		// 判断该特征点是否左目camera_id==0 ，这里不应该先检查然后添加？
        assert(id_pts.second[0].first == 0);
		// 如果是双目相机还需添加右目特征点
        if(id_pts.second.size() == 2)
        {
            f_per_fra.rightObservation(id_pts.second[1].second);
            assert(id_pts.second[1].first == 1);
        }

        int feature_id = id_pts.first;
		// 寻找feature容器中有无该特征点
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
		// 如果没有找到相同特征点就在容器中添加该特征点
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        }
		// 如果该特征点已经在之前出现过，就在其FeaturePerFrame内增加此特征点在此帧的位置以及其他信息
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if( it-> feature_per_frame.size() >= 4)
                long_track_num++;
        }
    }

    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
	// 2.通过视差或者特征点跟踪次数等条件判断当前帧是否是关键帧
	// 2.1 该帧识别的新特征点很多，说明该帧变化很大，返回true
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    for (auto &it_per_id : feature)
    {
		// 如果该特征点至少有两帧观测次数，且次新帧一定观测到了该特征点
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
			// 计算该特征点在上一帧和上上帧的归一化相机坐标系的视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
	// 说明所有点都是第一次观测到，认为该帧是关键帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
		// 2.2 该帧新特征点虽然不多，但是如果特征点在上一帧和上上帧中视差变化大，则也说明该帧变换很大，否则该帧变换很小
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
		// 如果次新帧和次次新帧之间的平均视差超过了规定阈值，认为关键帧，在滑窗中保留
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}
/**
 * @brief 找到两帧中的匹配特征点对
 * @param frame_count_l 图像1在滑动窗口的索引
 * @param frame_count_r 图像2的滑动窗口的索引
 * @return 返回图像1和2的匹配特征点对
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
		// 判断输入帧数是否在观测到该点的帧数范围内
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;
			// 找到该特征点在图像1和图像2上的位置
            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}
/**
 * @brief 设置特征点的深度
 * @param x 特征点的逆深度
 */
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
		// 将逆深度转为深度
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}
/**
 * @brief 把一些特征点深度估计出来是负的特征点都删除掉
 */
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;
}
/**
 * @brief 获取全部跟踪次数超过4次的特征点的逆深度向量
 * @return
 */
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
#if 1
		// 这里返回的是逆深度，也就是深度的倒数
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

/**
 * @brief 三角化得到特征点深度，原理参考：https://blog.csdn.net/qq_41904635/article/details/106092472
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param point_3d point3d是世界坐标系下的坐标
 */
void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * @brief PnP求解位姿
 * @param R [in,out]输入是w_R_cam，输出是w_R_cam
 * @param P [in,out]输入是w_t_cam，输出是w_t_cam
 * @param pts2D 特征点在cam（相机归一化坐标系下）的2D坐标
 * @param pts3D 特征点在世界坐标系下的3D坐标
 * @return
 */
bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
	// 由于pts2D是用相机归一化坐标系坐标代替像素坐标系下的坐标，因此相对于相机内参为单位阵
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
	// rvec和t是使坐标点从世界坐标系旋转到相机坐标系
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}
/**
 * @brief 通过上一帧的Ps[frameCnt-1]、Rs[frameCnt-1]以及当前帧看到了所有恢复了深度的2D-3D，通过PnP确定Ps[frameCnt]、Rs[frameCnt]
 * @param frameCnt 当前帧索引
 * @param Ps 滑动窗口中的相机位移 w_T_imu
 * @param Rs 滑动窗口中的相机姿态 w_T_imu
 * @param tic 相机的外参矩阵imu_t_cam，最多两个相机
 * @param ric 相机的外参矩阵imu_R_cam，最多两个相机
 */
void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{

    if(frameCnt > 0)
    {

        vector<cv::Point2f> pts2D;
		// 世界坐标系下
        vector<cv::Point3f> pts3D;
		// 遍历全部特征点，找到全部可以被该帧看到且已经估计出深度的特征点
        for (auto &it_per_id : feature)
        {
            if (it_per_id.estimated_depth > 0)
            {
				// 计算在数组中的索引值
                int index = frameCnt - it_per_id.start_frame;
				// pnp为了求当前的位姿，因此得保障当前这个特征点能被当前帧所看到
                if((int)it_per_id.feature_per_frame.size() >= index + 1)
                {
					// 把3D特征点从发现该特征点的第一帧的相机坐标系转到相应该帧的IMU坐标系下
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0];
                    // 然后再转到世界坐标系下
					Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d); 
                }
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
		// 用上一帧的位姿作为初值
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose(); 
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}
/**
 * @brief 恢复所有帧的特征点深度
 * @param frameCnt 当前帧的索引
 * @param Ps 滑动窗口内帧的w_t_imu
 * @param Rs 滑动窗口内帧的w_R_imu
 * @param tic 相机的外参矩阵imu_t_cam，最多两个相机
 * @param ric 相机的外参矩阵imu_R_cam，最多两个相机
 */
void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
	// 遍历全部特征点
    for (auto &it_per_id : feature)
    {
		// 如果该特征点深度>0，即该特征点的状态认为是好的，那么就跳过
        if (it_per_id.estimated_depth > 0)
            continue;
		// 如果是双目并且两个相机都能看到同一个点，那就根据两个相机的外参去进行三角化
        if(STEREO && it_per_id.feature_per_frame[0].is_stereo)
        {
            int imu_i = it_per_id.start_frame;
			// cam0_T_w
            Eigen::Matrix<double, 3, 4> leftPose;
			// w_T_cam
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
			// w_T_cam -> cam0_T_w
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            //cout << "left pose " << leftPose << endl;

			// cam1_T_w
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;
            //cout << "right pose " << rightPose << endl;

			// 左右目匹配的特征点
            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;

			// 三角化出点的深度，point3d是世界坐标系下的坐标
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
			// 在左目相机坐标系下的坐标
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
		// 如果是单目或者双目只有一个相机看到，那就需要同一个相机不同位姿的观测来三角化
        else if(it_per_id.feature_per_frame.size() > 1)
        {
            int imu_i = it_per_id.start_frame;
			// cam0_T_w
            Eigen::Matrix<double, 3, 4> leftPose;
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            imu_i++;
			// cam0_T_w
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[1].point.head(2);
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
		// 貌似下面就进不来了吧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}
/**
 * @brief 删除feature变量中outlierIndex索引的外点
 * @param outlierIndex 外点索引
 */
void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            feature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}
/**
 * @brief 把被移除帧(R0和P0)看见地图点的管理权交给当前的最老帧(R1和P1)
 * @param marg_R 被移除的位姿
 * @param marg_P
 * @param new_R 转接地图点的位姿
 * @param new_P
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
	// 遍历所有特征点
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
		// 如果遍历到的当前特征点it不是被移除的帧(第0帧)看到，那么该地图点对应的起始帧id减一(因为滑窗中所有帧的位姿都整体左移了一下)
        if (it->start_frame != 0)
            it->start_frame--;
        else
			// 如果遍历到的当前特征点被移除的帧(第0帧)所看到，则交接管辖权
        {
			// 取出当前特征点在原先第0帧归一化相机坐标系下的坐标
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
			// 该点不再被原来的第0帧看到，因此从中移除
            it->feature_per_frame.erase(it->feature_per_frame.begin());
			// 将从第0帧中移除后，如果这个地图点没有至少被两帧看到
            if (it->feature_per_frame.size() < 2)
            {
				// 那他就没有存在的价值了
                feature.erase(it);
                continue;
            }
            else
				// 说明当前特征点从原先第0帧中移除后，仍然至少被两帧所看到，表明它是一个有效特征点，下面进行管辖权交接（主要是更新一下特征点深度）
            {
				// 转换过程: 特征点在第0帧的归一化坐标 ==> 特征点在第0帧的相机坐标系下的坐标 ==> 特征点在世界坐标系下的坐标 ==> 特征点在第1帧的相机坐标系下的坐标
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}
/**
 * @brief 还未成功初始化时的特征点处理：直接移除掉对应特征点，因为此时未初始化成功还没有深度
 */
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
		// 如果遍历到的当前特征点it不是被移除的帧(第0帧)看到
		// 那么该地图点对应的起始帧id减一(因为滑窗中所有帧的位姿都整体左移了一下)
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
			// 起始帧为0时，直接将其移除掉
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}
/**
 * @brief 对margin倒数第二帧进行处理
 * @param frame_count
 */
void FeatureManager::removeFront(int frame_count)
{
	// 遍历所有特征点
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
		// 如果观测到地图点的起始帧是最后一帧，由于滑窗，他的起始帧减1(因为最后一帧会挪动，其他帧不会挪动)
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
			// 倒数第二帧能够观察到这个地图点
        {
			// 在观察到当前地图点的所有帧当中，倒数第二帧的索引值记为j
			// 比如假设WINDOW_SIZE = 10, 观察到地图点的起始帧在滑窗中的索引为5
			// 那么倒数第二帧在观察到该地图点的所有帧中的索引为10 - 1 - 5 = 4
            int j = WINDOW_SIZE - 1 - it->start_frame;
			// 如果能够观察到当前地图点的最后一帧索引 小于 倒数第二，则当前地图点不能被倒数第二帧所看到
            if (it->endFrame() < frame_count - 1)
                continue;
			// 如果当前地图点能被倒数第二帧所看到，则将倒数第二帧中该地图点删除掉
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
			// 如果这个地图点没有别的观测了
            if (it->feature_per_frame.size() == 0)
				// 就没有存在的价值了(为1的时候也没有存在的价值了)，把当前地图点也给删掉
                feature.erase(it);
        }
    }
}
/**
 * @brief 计算某特征点在观测到它的次新帧和次次新帧上的距离（归一化相机平面上）
 * @param it_per_id
 * @param frame_count
 * @return ** double
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}