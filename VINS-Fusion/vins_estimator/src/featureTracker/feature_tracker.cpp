/*******************************************************
该文件定义了特征点识别的函数
 *******************************************************/

#include "feature_tracker.h"
/**
 * @brief 判断特征点是否在图片边间内
 * @param pt 特征点
 * @return
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
	// 图片边框大小
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}
/**
 * @brief 计算两个特征点的欧氏距离
 * @param pt1 特征点1
 * @param pt2 特征点2
 * @return 两特征点的欧氏距离
 */
double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}
/**
 * @brief 剔除特征点向量中的外点，只保留特征点向量中跟踪成功的特征点
 * @param v 特征点队列，该函数会剔除队列中跟踪失败的点
 * @param status 队列，存储特征点是否跟踪成功
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}
/**
 * @brief 非极大值抑制筛选特征点，使特征点均匀化。将当前识别到的特征点，按照被追踪到的次数排序并依次选点，使用mask进行类似非极大抑制
 * 半径为MIN_DIST，去掉密集点，使特征点分布均匀。
 */
void FeatureTracker::setMask()
{
	// 创建和原图一样大的mask图片
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
	// key是跟踪次数，value是（特征点位置，特征点索引）
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));
	// 将特征点按照track_cnt的次数，从大到小排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
	// 非极大值抑制，相当于在一个圆区域中只保留跟踪次数最多的特征点
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
			// 如果特征点所在区域不在圆领域中就保留该特征点，并在该特征点附近画一个圆区域
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}
/**
 * @brief 计算两个特征点之间的欧氏距离
 * @param pt1 特征点1
 * @param pt2 特征点2
 * @return 特征点之间的欧氏距离
 */
double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}
/**
 * @brief 提取特征点，步骤为：
 * 1.用上一帧特征点正向光流法跟踪得到当前帧跟踪成功的特征点（反向光流法检查）
 * 2.非极大抑制，并提取新的角点特征点
 * 3.计算特征点在归一化相机坐标系的3D坐标和在图像坐标系的速度
 * 4.绘制跟踪图片
 * 5.保存结果
 * @param _cur_time 图片时间戳
 * @param _img 左目图片
 * @param _img1 右目图片，如果是单目相机则为空矩阵
 * @return feature_id：(camera_id：(x, y, z, p_u, p_v, velocity_x, velocity_y))
 * feature_id为特征点ID 、camera_id为相机ID（0为左目，1为右目）、(x, y, z)为特征点在归一化相机坐标系的3D坐标
 * (p_u, p_v)为特征点在图像坐标系的2D坐标、(velocity_x, velocity_y)为特征点在图像坐标系的速度
 *
 */
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    /*
    {
     	// 图片预处理，包括直方图均衡，这里源码注释了可能是怕影响后面光流法
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();
	// 1.如果上一帧有特征点（不是第一帧），就直接进行光流LK追踪当前帧的特征点
    if (prev_pts.size() > 0)
    {
        TicToc t_o;
		// status存储光流法跟踪的结果，1为成功，0为失败
        vector<uchar> status;
        vector<float> err;
		// 1.1 正向光流跟踪
        if(hasPrediction)
        {
			// 用预测模块预测的特征点位置作为光流法迭代初始值
            cur_pts = predict_pts;
			// 使用光流法对上一帧的特征点进行跟踪，得到当前帧的特征点位置
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            // 统计光流法跟踪成功的特征点数目
            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
			// 如果跟踪成功的数目太少，就调整图像金字塔层，再执行一次光流法跟踪
            if (succ_num < 10)
               cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }
		// 不使用预测模块，光流法不提供当前特征点的初始值
        else
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        // 1.2 反向光流法，提高光流法跟踪到的特征点匹配正确率
        if(FLOW_BACK)
        {
			// 反向光流即，用前面跟踪到的当前帧特征点，反向跟踪上一帧的特征点
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
            for(size_t i = 0; i < status.size(); i++)
            {
				// 只有正向光流和反向光流都成功，且反向光流追踪的误差小于阈值才认为该特征点被跟踪成功
                if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }
        // 1.3 剔除跟踪失败的外点
        for (int i = 0; i < int(cur_pts.size()); i++)
			// 超出图片边间的也认为跟踪失败
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
		// ids和track_cnt都只是记录当前帧成功跟踪到的特征点！！！
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }
	// 上一帧中跟踪成功的特征点，其跟踪计数都+1
    for (auto &n : track_cnt)
        n++;

	// 2.除了跟踪成功的旧特征点外，在当前帧中继续识别新的特征点
    if (1)
    {
		// 2.1 通过F剔除外点
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
		// 2.2 非极大值抑制筛选特征点，使特征点均匀化
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
		// 2.3 提取新的特征点
		// 计算还需要识别多少个新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
			// 在mask中不为0的区域检测新的特征点（Harris角点）
            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());
		// 把新提取的特征点加入到cur_pts、ids、track_cnt中
        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());
    }
	// 3.对识别到的特征点去畸变并归一化到相机平面
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
	// 4.计算当前帧特征点的速度
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
	// 5.如果是双目还需要对右目图片同样的操作，只不过右目特征点只会用左目特征点光流跟踪得到，不会提取新的特征点
    if(!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
			// 用左目特征点光流法跟踪右目特征点
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
			// 反向光流法检测
            if(FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
			// 如果左目的特征点右目没有，也没关系，不会影响左目已经提出来的特征点
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
	// 6.如果要显示跟踪图片，就绘制跟踪图片
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
	// 7.将当前帧的状态进行转移和保存，保存为函数输出结果
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];
	// 特征点ID：(相机ID：(x, y, z, p_u, p_v, velocity_x, velocity_y))
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
	// 左目信息保存
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
		// 归一化相机坐标系的3D坐标
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
		// 图片坐标系的2D坐标
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
		// 左目的camera_id = 0
        int camera_id = 0;
		// 图片坐标系上的速度
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
	// 右目信息保存
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
			// 右目的camera_id = 1
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}
/**
 * @brief 读取相机的内参文件并创建相机实例化对象
 * @param calib_file 队列，存储相机内参文件路径
 */
void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}
/**
 * @brief 对2D特征点去畸变，并变换到归一化坐标系下
 * @param pts 需要处理的2D特征点
 * @param cam 相机模型
 * @return 返回处理后的无畸变的归一化相机坐标
 */
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
		// liftProjective函数不是得到2D特征点对应的3D地图点，而是其投影射线点
		// 此外没看到去畸变的操作
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}
/**
 * @brief 计算两帧之间的点在归一化相机平面上的速度
 * @param ids 当前帧上特征点的ID号
 * @param pts 当前帧上特征点在归一化平面的3D坐标（仅包含x y）
 * @param cur_id_pts 字典 = 当前帧的（ids，pts）
 * @param prev_id_pts 字典 = 上一帧的（ids，pts）
 * @return pts中全部特征点在两帧中的XY速度 = ((特征点1的X速度，特征点1的Y速度)，(特征点2的X速度，特征点2的Y速度)，...)
 */
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
	// 通过当前归一化相机坐标位置（仅x, y）和角点ID重新赋值cur_id_pts
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // 计算上一帧和当前帧之间点的运动速度（或称为光流速度）
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
			// 找到两帧中同一特征点
            it = prev_id_pts.find(ids[i]);
			// 计算该特征点在两帧图片中的速度
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}
/**
 * @brief 绘制跟踪图片，也就是添加了特征点识别结果的图片
 * 对于左目特征点：跟踪次数少于20的，标记为蓝色点；跟踪次数大于20的，标记为红色点
 * 对于右目特征点：统一标记为绿色
 * 对于上一帧左目特征点中跟踪成功的特征点：使用绿色箭头绘制出来
 * @param imLeft 当前帧左目图片
 * @param imRight 当前帧右目图片
 * @param curLeftIds 当前帧左目特征点ID
 * @param curLeftPts 当前帧左目识别到的特征点
 * @param curRightPts 当前帧右目识别到的特征点
 * @param prevLeftPtsMap 上一帧图像左目的特征点和ID
 */
void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
		// 两图片水平拼接
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);
	// 绘制左目特征点
    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
		// 跟踪次数少于20的，标记为蓝色点；跟踪次数大于20的，标记为红色点
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
	// 绘制右目特征点
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
			// 右目特征点统一标记为绿色
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    // 绘制上一帧左目特征点中跟踪成功的特征点
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
		// 如果是上一帧左目特征点中跟踪成功的，就使用箭头绘制出来
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

/**
 * @brief 设置下一帧特征点的预测位置
 * @param predictPts 预测的特征点相机坐标系位置
 */
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}

/**
 * @brief 删除removePtsIds索引的外点
 * @param removePtsIds 外点索引
 */
void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}

/**
 * @brief 获取trackImage函数中绘制的跟踪图片
 * @return
 */
cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}