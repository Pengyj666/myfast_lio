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

#include "keyframe.h"

#include "common/log_filters.h"
#include "common/sysutils.h"

#include "droslog/log.h"
#include "geo_utils/geo_utils.h"

using namespace utils;

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

const cv::Mat Rm_rdf2flu = (cv::Mat_<float>(3, 3) << 0,  0, 1, 
                                                    -1,  0, 0, 
                                                     0, -1, 0);
const cv::Mat Rm_flu2rdf = (cv::Mat_<float>(3, 3) << 0, -1, 0, 
                                                     0, 0, -1, 
                                                     1, 0, 0);

cv::Point3f cam_flu2rdf(cv::Point3f cam_point) {
  return cv::Point3f(-cam_point.y, -cam_point.z, cam_point.x);
}


static int s_KF_cnt = 0;
std::mutex s_KF_cnt_mutex;
void add_KF_cnt() {
	std::lock_guard<std::mutex> lock(s_KF_cnt_mutex);
	s_KF_cnt++;
}
void sub_KF_cnt() {
	std::lock_guard<std::mutex> lock(s_KF_cnt_mutex);
	s_KF_cnt--;
}
int get_KF_cnt() {
	std::lock_guard<std::mutex> lock(s_KF_cnt_mutex);
	return s_KF_cnt;
}


// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence)
{
	add_KF_cnt();
	if (get_KF_cnt() % 5 == 0) {
		droslog(LogLevel::INFO, "KeyFrame::ctor() add new KF, cnt = %d", get_KF_cnt());
	}

	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	if (!_image.empty()) {
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint();
	computeBRIEFPoint();
	if(!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	add_KF_cnt();
	if (get_KF_cnt() % 5 == 0) {
		droslog(LogLevel::INFO, "KeyFrame::ctor() load new KF, cnt = %d", get_KF_cnt());
	}

	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE && !_image.empty())
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

KeyFrame::~KeyFrame() {
	sub_KF_cnt();
	int cnt = get_KF_cnt();
	
	// 2025-12-04: 析构时打印内存地址信息，用于验证内存释放
	if (cnt == 0) {
		droslog(LogLevel::INFO, "KeyFrame::dtor() 销毁了所有KF, cnt = 0");

		
	} else if (cnt % 50 == 0) {
		droslog(LogLevel::INFO, "KeyFrame::dtor() delete KF, cnt = %d", cnt);
	}
}

void KeyFrame::SetRefLocInfo(const RefLocInfo &ref_loc_info) {
	ref_loc_info_ = ref_loc_info;
}

void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 20; // corner detector response threshold
	if(1)
		cv::FAST(image, keypoints, fast_th, true);
	else
	{
		vector<cv::Point2f> tmp_pts;
		cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
		for(int i = 0; i < (int)tmp_pts.size(); i++)
		{
		    cv::KeyPoint key;
		    key.pt = tmp_pts[i];
		    keypoints.push_back(key);
		}
	}
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
      best_match = keypoints_old[bestIndex].pt;
      best_match_norm = keypoints_old_norm[bestIndex].pt;
      return true;
    }
    else
      return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}

/* 输入：
descriptors_old               keypoints_old / keypoints_old_norm          predicted_R_old / predicted_T_old
旧关键帧的 BRIEF 描述子          旧关键帧的特征点像素坐标与归一化坐标             预测当前帧相对于 old keyframe 的位姿（来自 VIO 预测）

window_brief_descriptors（隐形成员变量）                       point_3d[i]（隐形成员）
当前关键帧所有 BRIEF 描述子（要去 old frame 中找匹配）            当前帧特征点的 3D 位置（若已有深度）


输出“
matched_2d_old                                             matched_2d_old_norm
当前关键帧每个特征点在 old keyframe 中匹配到的 2D 像素点          匹配到的归一化坐标

status
1 = 匹配成功，0 = 匹配失败 */
// 新增：基于预测位姿的引导匹配 - 2025-11-27
// 功能：利用VIO位姿预测特征点在目标帧中的位置，缩小搜索范围，提高匹配精度和速度
// 修复：2025-11-27 - 全面修正（输入验证、相机内参、去重策略、深度检查、空间约束、性能优化）
void KeyFrame::searchByBRIEFDesWithPoseGuide(std::vector<cv::Point2f> &matched_2d_old,
                                              std::vector<cv::Point2f> &matched_2d_old_norm,
                                              std::vector<uchar> &status,  // 匹配状态
                                              const std::vector<BRIEF::bitset> &descriptors_old,
                                              const std::vector<cv::KeyPoint> &keypoints_old, 
                                              const std::vector<cv::KeyPoint> &keypoints_old_norm,  
                                              const Eigen::Vector3d &predicted_T_old,  
                                              const Eigen::Matrix3d &predicted_R_old,  
                                              double search_radius)  // 搜索半径
{
    // 输入验证：防止越界和空指针 - 2025-11-27
    if (descriptors_old.empty() || keypoints_old.empty() || 
        keypoints_old.size() != descriptors_old.size() ||
        keypoints_old_norm.size() != keypoints_old.size()) {
        droslog(LogLevel::WARN, "KeyFrame::searchByBRIEFDesWithPoseGuide() 输入数据不一致，跳过");
        // 填充空结果
        for(int i = 0; i < (int)window_brief_descriptors.size(); i++) { 
            matched_2d_old.push_back(cv::Point2f(-1.f, -1.f));
            matched_2d_old_norm.push_back(cv::Point2f(-1.f, -1.f));
        }
        return;
    }
    // 预计算搜索半径的平方，避免重复sqrt计算 - 2025-11-27
    double search_radius_sq = search_radius * search_radius;
    
    // fallback 半径根据 search_radius 动态计算 - 2026-01-07
    // 引导匹配失败时使用更大的搜索半径进行全局匹配兜底
    // fallback_radius = search_radius + 80（保持合理的扩展范围）
    double fallback_radius = search_radius + 80.0;  // 例如：70+80=150, 150+80=230
    double fallback_radius_sq = fallback_radius * fallback_radius;
    
    // 修复问题2：仅在引导匹配阶段使用去重，fallback阶段不去重以提高匹配数量 - 2025-11-27
    std::vector<bool> matched_old_guided(descriptors_old.size(), false);
    
    int guided_match_cnt = 0;    // 引导匹配成功数量
    int fallback_match_cnt = 0;  // 全局搜索成功数量
    int no_depth_cnt = 0;        // 无深度点数量
    
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(-1.f, -1.f);        // 修复问题5：未匹配点初始化为(-1,-1) - 2025-11-27
        cv::Point2f pt_norm(-1.f, -1.f);
        bool matched = false;
        double predicted_u = -1.0;
        double predicted_v = -1.0;
        bool has_valid_projection = false;
        
        // 修复问题4：检查当前点是否有有效 3D 坐标 - 2025-11-27 / 2025-12-30 修正
        // 注意：point_3d 是世界坐标系（AS/FLU 坐标系）的点，z 是高度而非深度
        // 检查点是否有效：世界坐标不为零（VIO 发布时已确保 solve_flag=1）
        bool has_valid_3d = (i < (int)point_3d.size()) && 
                            (std::abs(point_3d[i].x) > 0.01 || 
                             std::abs(point_3d[i].y) > 0.01 || 
                             std::abs(point_3d[i].z) > 0.01);
        if (has_valid_3d) {
            // 获取当前帧的3D点（世界坐标系 FLU）
            Eigen::Vector3d P_world(point_3d[i].x, point_3d[i].y, point_3d[i].z);
            
            // 世界坐标系 → 旧帧 IMU 坐标系 → 旧帧相机坐标系
            Eigen::Vector3d P_old_imu = predicted_R_old.transpose() * (P_world - predicted_T_old);
            Eigen::Vector3d P_old_cam = qic.transpose() * (P_old_imu - tic);
            
            // 投影到old_kf的图像平面（P_old_cam.z() 是相机坐标系深度）
            if (P_old_cam.z() > 0.1) { // 深度有效性检查：在相机前方
                Eigen::Vector2d predicted_uv_norm(P_old_cam.x() / P_old_cam.z(), P_old_cam.y() / P_old_cam.z());
                
                // 转换到像素坐标（使用固定焦距，与PnP中保持一致）- 2025-11-27
                double FOCAL_LENGTH = 460.0;  // 与FundmantalMatrixRANSAC中一致
                predicted_u = predicted_uv_norm.x() * FOCAL_LENGTH + COL / 2.0;
                predicted_v = predicted_uv_norm.y() * FOCAL_LENGTH + ROW / 2.0;
                
                // 边界检查，预测位置必须在图像范围内
                if (predicted_u >= 0 && predicted_u < COL && 
                    predicted_v >= 0 && predicted_v < ROW) {
                    has_valid_projection = true;
                    
                    // 阶段1：引导匹配（小半径 + 去重）
                    int bestDist = 128;
                    int bestIndex = -1;
                    
                    for(int j = 0; j < (int)descriptors_old.size(); j++)
                    {
                        // 引导匹配阶段使用去重，保证高质量匹配
                        if (matched_old_guided[j]) continue;
                        
                        // 性能优化：使用平方距离避免sqrt - 2025-11-27
                        double dx = keypoints_old[j].pt.x - predicted_u;
                        double dy = keypoints_old[j].pt.y - predicted_v;
                        double dist_pixel_sq = dx * dx + dy * dy;
                        
                        if (dist_pixel_sq < search_radius_sq) {
                            int dis = HammingDis(window_brief_descriptors[i], descriptors_old[j]);
                            if(dis < bestDist)
                            {
                                bestDist = dis;
                                bestIndex = j;
                            }
                        }
                    }
                    
                    // 使用原始Hamming阈值80（保持一致）
                    if (bestIndex != -1 && bestDist < 80) {
                        pt = keypoints_old[bestIndex].pt;
                        pt_norm = keypoints_old_norm[bestIndex].pt;
                        matched = true;
                        matched_old_guided[bestIndex] = true;  // 标记为已匹配
                        guided_match_cnt++;
                    }
                }
            }
        } else {
            no_depth_cnt++;
        }
        
        // 阶段2：如果引导匹配失败，退回到全局搜索（保证鲁棒性）
        // 修复问题2：fallback阶段不使用去重，允许one-to-many以提高匹配数量 - 2025-11-27
        // 修复问题3：fallback添加空间约束，避免错误匹配 - 2025-11-27
        if (!matched) {
            int bestDist = 128;
            int bestIndex = -1;
            
            for(int j = 0; j < (int)descriptors_old.size(); j++)
            {
                // 修复问题3：fallback阶段添加空间约束 - 2025-11-27
                // 如果有有效投影，使用大半径空间约束；否则全局搜索
                if (has_valid_projection) {
                    // 性能优化：使用平方距离避免sqrt - 2025-11-27
                    double dx = keypoints_old[j].pt.x - predicted_u;
                    double dy = keypoints_old[j].pt.y - predicted_v;
                    double dist_pixel_sq = dx * dx + dy * dy;
                    
                    // fallback阶段使用更大的搜索半径，但仍有空间约束
                    if (dist_pixel_sq > fallback_radius_sq) continue;
                }
                
                int dis = HammingDis(window_brief_descriptors[i], descriptors_old[j]);
                if(dis < bestDist)
                {
                    bestDist = dis;
                    bestIndex = j;
                }
            }
            
            // 使用原始Hamming阈值80
            if (bestIndex != -1 && bestDist < 80) {
                pt = keypoints_old[bestIndex].pt;
                pt_norm = keypoints_old_norm[bestIndex].pt;
                matched = true;
                // fallback阶段不标记matched_old，允许重复匹配
                fallback_match_cnt++;
            }
        }
        
        if (matched) {
            status.push_back(1);
        } else {
            status.push_back(0);
        }
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
    
    // 匹配统计信息：只在成功时输出，每5秒最多1次，避免刷屏
    static SimpleLogFilter match_log_filter(5000);
    static int match_total_cnt = 0;
    static int match_success_cnt = 0;
    match_total_cnt++;
    if (guided_match_cnt + fallback_match_cnt > 0) {
        match_success_cnt++;
    }
    if (match_log_filter.Output(GetNow_Steady())) {
        droslog(LogLevel::INFO, "KeyFrame::searchByBRIEFDesWithPoseGuide() 统计: 成功率=%d/%d (%.1f%%), 本次: 引导=%d, 全局=%d, 无深度=%d", 
                match_success_cnt, match_total_cnt, 
                match_total_cnt > 0 ? 100.0 * match_success_cnt / match_total_cnt : 0.0,
                guided_match_cnt, fallback_match_cnt, no_depth_cnt);
    }
}


void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

// ========== VIO 运行时间管理 - 2026-01-07 ==========
// 用于动态调整引导匹配半径和 PnP 验证策略
namespace {
	static long long s_vio_start_time = 0;
	
	// 获取 VIO 运行时长（毫秒）
	long long getVioRunTimeMs() {
		if (s_vio_start_time == 0) {
			s_vio_start_time = GetNow_Steady();
		}
		return GetNow_Steady() - s_vio_start_time;
	}
	
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	std::vector<cv::Point3f> matched_3d_rdf;
	for (size_t i = 0; i < matched_3d.size(); i++)
	{
		matched_3d_rdf.push_back(cam_flu2rdf(matched_3d[i]));
	}
	
	cv::Mat r, rvec, t, D, tmp_r;
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	Matrix3d R_inital;
	Vector3d P_inital;
	Matrix3d R_w_c = origin_vio_R * qic;
	Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

	// 生成pnp解算初始值
	R_inital = R_w_c.inverse();
	P_inital = -(R_inital * T_w_c);

	cv::eigen2cv(R_inital, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_inital, t);

	cv::Mat inliers;
	TicToc t_pnp_ransac;

	// PnP RANSAC 求解
	// 参数说明：
	//   - iterationsCount=100: RANSAC 迭代次数
	//   - reprojectionError=20/460: 重投影误差阈值（约 20 像素，割草机场景放宽）
	//   - confidence=0.99: 置信度
	const double reproj_error = 20.0 / 460.0;
	bool pnp_ret = solvePnPRansac(matched_3d_rdf, matched_2d_old_norm, K, D, rvec, t, 
	                               true, 100, reproj_error, 0.99, inliers);

	for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
			status.push_back(0);

	for( int i = 0; i < inliers.rows; i++)
	{
			int n = inliers.at<int>(i);
			status[n] = 1;
	}

	// pnp解算位姿转换
	cv::Rodrigues(rvec, r);
	cv::Mat reloc_Rcw = (cv::Mat_<float>(3, 3) << r.at<double>(0, 0), r.at<double>(0, 1), r.at<double>(0, 2),
																							r.at<double>(1, 0), r.at<double>(1, 1), r.at<double>(1, 2),
																							r.at<double>(2, 0), r.at<double>(2, 1), r.at<double>(2, 2));
	cv::Mat reloc_tcw = (cv::Mat_<float>(3, 1) << t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
	
	cv::Mat reloc_Rwc = reloc_Rcw.t();
	cv::Mat reloc_twc = -reloc_Rwc * reloc_tcw;

	reloc_Rwc = Rm_rdf2flu * reloc_Rwc * Rm_rdf2flu.t();
	reloc_twc = Rm_rdf2flu * reloc_twc;

	Matrix3d R_w_c_old;
	Vector3d T_w_c_old;
	cv::cv2eigen(reloc_Rwc, R_w_c_old);
	cv::cv2eigen(reloc_twc, T_w_c_old);

	// pnp解算位姿转换到世界坐标系
	PnP_R_old = R_w_c_old * qic.transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * tic;

	// ========== 修改 - 2026-01-07 ==========
	// 取消距离阈值过滤，只依靠 RANSAC 内点比例验证
	// 原因：割草机场景 VIO 漂移可能较大，距离阈值会误拒绝正确结果
	// GPS 粗验证在 findConnection() 中单独处理
	int inlier_cnt = inliers.rows;
	double inlier_ratio = (matched_3d.size() > 0) ? (double)inlier_cnt / matched_3d.size() : 0.0;
	double pos_diff = (PnP_T_old - origin_vio_T).norm();
	
	// 统计变量（用于降频日志）
	static int s_total_cnt = 0;
	static int s_accept_cnt = 0;
	static double s_max_diff = 0.0;
	s_total_cnt++;
	
	if (!pnp_ret) {
		// PnP 求解失败（OpenCV 返回 false）
		status.clear();
		for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) {
			status.push_back(0);
		}
		
		static SimpleLogFilter fail_filter(10000);  // 10秒一次
		if (fail_filter.Output(GetNow_Steady())) {
			droslog(LogLevel::WARN, "PnPRANSAC 求解失败: matched=%d", (int)matched_3d.size());
		}
	} else {
		// PnP 成功，不做距离阈值过滤，由 RANSAC 内点比例控制质量
		s_accept_cnt++;
		s_max_diff = std::max(s_max_diff, pos_diff);
		
		// 降频日志：每10秒输出一次统计
		static SimpleLogFilter accept_filter(10000);
		if (accept_filter.Output(GetNow_Steady())) {
			droslog(LogLevel::INFO, "PnPRANSAC 统计: 成功=%d/%d, 最大偏差=%.1fm, 本次: inlier=%d/%.0f%%, diff=%.1fm", 
					s_accept_cnt, s_total_cnt, s_max_diff, 
					inlier_cnt, inlier_ratio * 100.0, pos_diff);
			s_max_diff = 0.0;  // 重置最大偏差
		}
	}
}

namespace {

/// 将角度保持在正负PI以内
double KeepAngleInPI(const double& _angle) {
	double angle = _angle;
	while (angle < -M_PI) {
			angle = angle + 2 * M_PI;
	}
	while (angle > M_PI) {
			angle = angle - 2 * M_PI;
	}
	return angle;
}
	
} // namespace 

bool KeyFrame::findConnection(KeyFrame* old_kf, bool is_first_reloc)
{
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)    
		{
			cv::Mat gray_img, loop_match_img;
			cv::Mat old_img = old_kf->image;
			cv::hconcat(image, old_img, gray_img);
			cv::cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
			for(int i = 0; i< (int)point_2d_uv.size(); i++)
			{
					cv::Point2f cur_pt = point_2d_uv[i];
					cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
			}
			for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
			{
					cv::Point2f old_pt = old_kf->keypoints[i].pt;
					old_pt.x += COL;
					cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 0, 255));
			}
			ostringstream path;
			path << "reloc_imgs/"
							<< index << "-"
							<< old_kf->index << "-" << "0raw_point.jpg";
			cv::imwrite(path.str(), loop_match_img);
		}
	#endif
	//printf("search by des\n");
	
	// ========== 全局 BRIEF 匹配 - 2026-01-12 ==========
	// 不使用引导匹配，因为 VIO 漂移可能导致引导位置错误
	// 全局搜索虽然慢一点，但更可靠
	searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, 
	                 old_kf->brief_descriptors, old_kf->keypoints, 
	                 old_kf->keypoints_norm);
	
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	//printf("search by des finish\n");
	int matched_keypoints_num = matched_2d_cur.size();

	#if 0
		if (DEBUG_IMAGE)
		{
			int gap = 10;
			cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
			cv::Mat gray_img, loop_match_img;
			cv::Mat old_img = old_kf->image;
			cv::hconcat(image, gap_image, gap_image);
			cv::hconcat(gap_image, old_img, gray_img);
			cv::cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
			for(int i = 0; i< (int)matched_2d_cur.size(); i++)
			{
					cv::Point2f cur_pt = matched_2d_cur[i];
					cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
			}
			for(int i = 0; i< (int)matched_2d_old.size(); i++)
			{
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x += (COL + gap);
					cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 0, 255));
			}
			for (int i = 0; i< (int)matched_2d_cur.size(); i++)
			{
					cv::Point2f old_pt = matched_2d_old[i];
					old_pt.x +=  (COL + gap);
					cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
			}

			ostringstream path, path1, path2;
			path <<  "reloc_imgs/"
							<< index << "-"
							<< old_kf->index << "-" << "1descriptor_match.jpg";
			cv::imwrite( path.str().c_str(), loop_match_img);
			/*
			path1 <<  "/home/tony-ws1/raw_data/loop_image/"
							<< index << "-"
							<< old_kf->index << "-" << "1descriptor_match_1.jpg";
			cv::imwrite( path1.str().c_str(), image);
			path2 <<  "/home/tony-ws1/raw_data/loop_image/"
							<< index << "-"
							<< old_kf->index << "-" << "1descriptor_match_2.jpg";
			cv::imwrite( path2.str().c_str(), old_img);	        
			*/
		}
	#endif
	status.clear();
	/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
		PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		reduceVector(matched_2d_cur, status);
		reduceVector(matched_2d_old, status);
		reduceVector(matched_2d_cur_norm, status);
		reduceVector(matched_2d_old_norm, status);
		reduceVector(matched_3d, status);
		reduceVector(matched_id, status);
	#if 1
		if (DEBUG_IMAGE && !image.empty() && !old_kf->image.empty())
		{
			int gap = 10;
			cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
			cv::Mat gray_img, loop_match_img;
			cv::Mat old_img = old_kf->image;
			
			// 确保图像尺寸一致，避免 hconcat 崩溃
			cv::Mat cur_img = image;
			if (cur_img.rows != old_img.rows) {
				// 调整到相同高度
				int target_rows = std::min(cur_img.rows, old_img.rows);
				if (cur_img.rows != target_rows) {
					cv::resize(cur_img, cur_img, cv::Size(cur_img.cols * target_rows / cur_img.rows, target_rows));
				}
				if (old_img.rows != target_rows) {
					cv::resize(old_img, old_img, cv::Size(old_img.cols * target_rows / old_img.rows, target_rows));
				}
				gap_image = cv::Mat(target_rows, gap, CV_8UC1, cv::Scalar(255, 255, 255));
			}
			
			cv::hconcat(cur_img, gap_image, gap_image);
			cv::hconcat(gap_image, old_img, gray_img);
			cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
				for(int i = 0; i< (int)matched_2d_cur.size(); i++)
				{
						cv::Point2f cur_pt = matched_2d_cur[i];
						cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
				}
				for(int i = 0; i< (int)matched_2d_old.size(); i++)
				{
						cv::Point2f old_pt = matched_2d_old[i];
						old_pt.x += (COL + gap);
						cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
				}
				for (int i = 0; i< (int)matched_2d_cur.size(); i++)
				{
						cv::Point2f old_pt = matched_2d_old[i];
						old_pt.x += (COL + gap) ;
						cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
				}
				cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
				putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

				putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
				cv::vconcat(notation, loop_match_img, loop_match_img);

				/*
				ostringstream path;
				path <<  "/home/tony-ws1/raw_data/loop_image/"
								<< index << "-"
								<< old_kf->index << "-" << "3pnp_match.jpg";
				cv::imwrite( path.str().c_str(), loop_match_img);
				*/
				if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
				{
					/*
					cv::imshow("loop connection",loop_match_img);  
					cv::waitKey(10);  
					*/
					cv::Mat thumbimage;
					cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
					sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
					msg->header.stamp = ros::Time(time_stamp);
					pub_match_img.publish(msg);
				}
			}
		#endif
	}
	
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		// ========== 相对位姿计算 ==========
		// 
		// relative_t = PnP_R^T * (VIO_T - PnP_T)
		// 含义：VIO 位姿与 PnP 位姿的偏差（在 PnP 坐标系下）
		// 作用：当 VIO 漂移小时，说明 VIO 还比较准，PnP 结果可信
		//       当 VIO 漂移大时，说明可能是误匹配，需要 GPS 验证
		// 
		// 这种方式能有效屏蔽误匹配：
		//   - 正确匹配：PnP 位姿 ≈ 真实位姿，VIO 漂移 = |VIO - 真实| 
		//   - 错误匹配：PnP 位姿偏离真实位置，VIO 漂移会很大
		
		// 计算 VIO 与 PnP 的偏差（原始方法）
		relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
		relative_q = Eigen::Quaterniond(PnP_R_old.transpose() * origin_vio_R);
		relative_yaw = KeepAngleInPI(GetEulerRPY(origin_vio_R)[2] - GetEulerRPY(PnP_R_old)[2]);
		
		// VIO-PnP 偏差（用于日志）
		// 注意：这不是真正的 VIO 漂移，而是 VIO 位姿与 PnP 解算位姿的偏差
		// 真正的 VIO 漂移在 fusion_localization 的 off_rtk_dist 日志中
		double vio_pnp_diff = relative_t.norm();
		
		// ========== GPS 验证 - 2026-01-14 只使用 RTK 固定解 ==========
		// GPS 类型：0-充电桩, 1-RTK_NARROW_INT(固定解), 2-RTK_NARROW_FLOAT(浮点解), 3-RTK_SINGLE(单点解), -1-无效
		// 只使用固定解(type=0,1)做验证，浮点解/单点解/无效数据不参与验证
		bool gps_check_passed = true;
		double gps_diff = -1.0;
		double gps_threshold = 3.0;
		
		// 只有充电桩(0)和RTK固定解(1)才做GPS验证
		if (ref_loc_info_.type == 0 || ref_loc_info_.type == 1) {
			Eigen::Vector3d gps_pos = ref_loc_info_.xyz;
			gps_diff = (PnP_T_old - gps_pos).head<2>().norm();  // 只比较 XY 平面
			
			// 充电桩阈值更严格
			gps_threshold = (ref_loc_info_.type == 0) ? 2.0 : 3.0;
			
			if (gps_diff > gps_threshold) {
				gps_check_passed = false;
				
				// 降频日志：每10秒输出一次
				static SimpleLogFilter gps_fail_filter(10000);
				if (gps_fail_filter.Output(GetNow_Steady())) {
					droslog(LogLevel::WARN, "findConnection() GPS验证失败: PnP=(%.1f,%.1f), GPS=(%.1f,%.1f), 偏差=%.1fm>%.0fm, type=%d", 
							PnP_T_old[0], PnP_T_old[1], gps_pos[0], gps_pos[1], gps_diff, gps_threshold, ref_loc_info_.type);
				}
			}
		}
		// 注：浮点解(2)/单点解(3)/无效(-1)时，不做GPS验证，完全依赖视觉匹配
		
		// ========== 验证逻辑 - 2026-01-13 完全参考 VioTracker 修改 ==========

		//   1. 首次重定位时不做任何验证，直接让 PnP 结果 feed 到 VioTracker
		//   2. 由 VioTracker 的 spa_align 来判断是否可信
		//   3. 只做基本的姿态异常检测（roll/pitch）
		//   4. 已对齐后才使用完整验证（GPS + 角度 + relative_t）
		
		bool validation_passed = false;
		
		if (is_first_reloc) {
			// ========== 首次重定位：不做 GPS 粗验证 ==========
			// 完全参考 VioTracker：让 PnP 结果直接 feed 到 VioTracker
			// 由 spa_align 图优化来判断是否可信
			// 
			// 只做基本的姿态异常检测（参考 VioTracker 第353行）
			// VioTracker 检查 roll/pitch：|roll| > 0.3 || |pitch| > 0.3
			// 这里检查 yaw 的合理性（yaw 在两个坐标系中应该一致）
			auto pnp_rpy = GetEulerRPY(PnP_R_old);
			if (std::abs(pnp_rpy[0]) > 0.3 || std::abs(pnp_rpy[1]) > 0.3) {
				// PnP 解算的 roll/pitch 异常（割草机应该接近水平）
				static SimpleLogFilter rp_fail_filter(5000);
				if (rp_fail_filter.Output(GetNow_Steady())) {
					droslog(LogLevel::WARN, "findConnection() 首次重定位姿态异常: roll=%.2f, pitch=%.2f", 
							pnp_rpy[0], pnp_rpy[1]);
				}
			} else {
				// 首次重定位：不做其他验证，直接通过
				// 让 VioTracker 的 spa_align 来判断是否可信
				validation_passed = true;
				droslog(LogLevel::INFO, "findConnection() 首次重定位(待spa_align验证): kp=%d, 3d=%d, pnp_pos=(%.2f,%.2f,%.2f), gps_diff=%.1fm", 
						matched_keypoints_num, (int)matched_3d.size(), 
						PnP_T_old[0], PnP_T_old[1], PnP_T_old[2], gps_diff);
			}
		} else {
			// ========== 已对齐后的验证 ==========
			// VioTracker 已经完成 spa_align，VIO 坐标系已对齐到地图坐标系
			// 此时 relative_t 验证有效，使用完整验证
			// 2026-01-11: 阈值 0.5m，避免 RTK 丢失时漂移过快
			if (gps_check_passed && std::abs(relative_yaw) < 0.3 && relative_t.norm() < 0.5) {
				validation_passed = true;
				if (gps_diff >= 0) {
					droslog(LogLevel::INFO, "findConnection() 成功: kp=%d, 3d=%d, rel_t=(%.2f,%.2f,%.2f), gps_diff=%.1fm, vio_pnp_diff=%.2fm", 
							matched_keypoints_num, (int)matched_3d.size(), relative_t[0], relative_t[1], relative_t[2], gps_diff, vio_pnp_diff);
				} else {
					droslog(LogLevel::INFO, "findConnection() 成功(无GPS): kp=%d, 3d=%d, rel_t=(%.2f,%.2f,%.2f), vio_pnp_diff=%.2fm", 
							matched_keypoints_num, (int)matched_3d.size(), relative_t[0], relative_t[1], relative_t[2], vio_pnp_diff);
				}
			}
		}
		
		if (validation_passed) {
			has_loop = true;
			loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
										relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
										relative_yaw;
			return true;
		}
		// PnP 验证失败（距离/角度过大 或 GPS 不一致），不输出日志避免刷屏
	}
	
	// 匹配失败，使用静态计数器统计，每10秒输出一次汇总
	static int fail_count = 0;
	static double last_fail_log_time = 0.0;
	fail_count++;
	double now = GetNow_Steady();
	if (now - last_fail_log_time > 10000.0) {
		droslog(LogLevel::WARN, "KeyFrame::findConnection() 失败统计: 最近10秒失败 %d 次", fail_count);
		fail_count = 0;
		last_fail_log_time = now;
	}
	return false;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


