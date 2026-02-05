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

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 20

using namespace Eigen;
using namespace std;
using namespace DVision;

// KeyFrame 全局计数器函数声明 - 2025-12-04
int get_KF_cnt();

class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

struct RefLocInfo {
  int type = -1; // -1-unknown, 0-On_Charge_Station, 1-RTK_NARROW_INT, 2-RTK_NARROW_FLOAT, 3-RTK_SINGLE
	double timestamp = 0.0;
	Vector3d xyz = Vector3d::Zero();
	Matrix3d cov = Matrix3d::Identity();
};

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	
	~KeyFrame();

	void SetRefLocInfo(const RefLocInfo &ref_loc_info);
  
	// is_first_reloc: 是否是首次重定位（VIO 坐标系还未对齐到地图坐标系）
	// 参考 VioTracker 的思路：首次重定位时跳过 relative_t 验证，只依赖 GPS 验证
	// 因为 relative_t = PnP_R^T * (VIO_T - PnP_T)，VIO_T 在 VIO 坐标系，PnP_T 在地图坐标系
	// 首次重定位时两者差值可能有几米甚至十几米，不能用于验证
	bool findConnection(KeyFrame* old_kf, bool is_first_reloc = false);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();

	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	// 新增：基于预测位姿的引导匹配 - 2025-11-27
	void searchByBRIEFDesWithPoseGuide(std::vector<cv::Point2f> &matched_2d_old,
	                                    std::vector<cv::Point2f> &matched_2d_old_norm,
	                                    std::vector<uchar> &status,
	                                    const std::vector<BRIEF::bitset> &descriptors_old,
	                                    const std::vector<cv::KeyPoint> &keypoints_old,
	                                    const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                                    const Eigen::Vector3d &predicted_T_old,
	                                    const Eigen::Matrix3d &predicted_R_old,
	                                    double search_radius);
										
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();


	double time_stamp; 
	int index;
	int local_index;
	Eigen::Vector3d vio_T_w_i; 	// 当前VIO转map
	Eigen::Matrix3d vio_R_w_i;
	Eigen::Vector3d T_w_i;			// map坐标系
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d origin_vio_T;	// VIO原始位姿
	Eigen::Matrix3d origin_vio_R;
	
	cv::Mat image;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d; 
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
	vector<double> point_id;
	vector<cv::KeyPoint> keypoints;
	vector<cv::KeyPoint> keypoints_norm;
	vector<cv::KeyPoint> window_keypoints;
	vector<BRIEF::bitset> brief_descriptors;
	vector<BRIEF::bitset> window_brief_descriptors;
	bool has_fast_point;
	int sequence;

	RefLocInfo ref_loc_info_;

	bool has_loop;
	int loop_index;
	Eigen::Matrix<double, 8, 1 > loop_info;

	// ========== 空间索引信息（由 SpatialMapManager 填充）- 2025-12-10 ==========
	int submap_x = 0;        // 子图 X 坐标 (floor(x/5.0))
	int submap_y = 0;        // 子图 Y 坐标 (floor(y/5.0))
	int cell_x = 0;          // Cell X 坐标 (floor(x/0.25))
	int cell_y = 0;          // Cell Y 坐标 (floor(y/0.25))
	int direction_slot = 0;  // 方向槽位 (0-5)，每60度一个槽位
	
	// ========== 增量索引更新支持 - 2025-12-15 ==========
	// 缓存的索引位置（用于检测位姿优化后是否需要重新索引）
	int cached_submap_x = 0;
	int cached_submap_y = 0;
	int cached_cell_x = 0;
	int cached_cell_y = 0;
	int cached_direction_slot = 0;
	bool index_dirty = false;         // 位姿已更新但索引未更新
	bool is_segment_optimized = false; // 是否已被段优化处理过
	
	// ========== 优化质量标记 - 2025-12-23 ==========
	int optimization_quality = -1;    // OptimizationQuality 枚举值
};

