#include "simple_pose_graph.h"
#include "spatial_map_manager.h"

#include <stdio.h>
#include <fstream>
#include <set>
#include <algorithm>

#include "common/log_filters.h"
#include "common/sysutils.h"
#include "droslog/log.h"
#include "geo_utils/geo_utils.h"
#include "geo_utils/tf_helper.h"

#include "spa_align.h"
#include "parameters.h"
#include "utility/map_drawer.h"

using namespace utils;

SimplePoseGraph::SimplePoseGraph(int type) : type_(type), map_stage_(0)
{
  droslog(LogLevel::INFO, "SimplePoseGraph::ctor() ++++++");
  global_index = 0;
  droslog(LogLevel::INFO, "SimplePoseGraph::ctor() ------");
}

SimplePoseGraph::~SimplePoseGraph()
{
  droslog(LogLevel::INFO, "SimplePoseGraph::dtor() ++++++");
  
  // 先清空词袋数据库（释放大量内存）- 2025-12-01
  if (db_) {
    db_->clear();
  }
  db_.reset();
  voc_.reset();
  
  // 清空关键帧列表
  std::lock_guard<std::mutex> lock(keyframelist_mutex_);
  
  // 2025-12-04: 记录析构前的 KeyFrame 数量
  size_t kf_count_before = keyframelist_.size();
  droslog(LogLevel::INFO, "SimplePoseGraph::dtor() 开始析构 %zu 个 KeyFrame", kf_count_before);
  
  for (auto &kf : keyframelist_) {  
    kf.reset();
  }
  keyframelist_.clear();
  
  // 验证：检查全局 KF 计数是否为 0
  int remaining_kf = get_KF_cnt();
  if (remaining_kf == 0) {
    droslog(LogLevel::INFO, "SimplePoseGraph::dtor() 所有 KeyFrame 已析构完成 ✓");
  } else {
    droslog(LogLevel::WARN, "SimplePoseGraph::dtor() 警告：仍有 %d 个 KeyFrame 未析构！", remaining_kf);
  }
  
  droslog(LogLevel::INFO, "SimplePoseGraph::dtor() ------");
}

void SimplePoseGraph::loadVocabulary(std::string voc_path)  
{
  // 如果词汇表已加载且路径相同，只清空数据库，不重新加载词汇表
  if (voc_ && voc_path_ == voc_path) {
    droslog(LogLevel::INFO, "SimplePoseGraph::loadVocabulary() 词汇表已加载，跳过重复加载");
    clearDatabase();
    return;
  }
  
  voc_path_ = voc_path;
  
  // 先清空旧数据库的所有内容
  if (db_) {
    db_->clear();
  }
  db_.reset();
  
  // 2026-01-11: 清空 EntryId 映射表
  {
    std::lock_guard<std::mutex> lock(entry_map_mutex_);
    entry_to_kf_index_.clear();
  }
  
  // 加载词汇表（只在首次或路径变化时执行）
  droslog(LogLevel::INFO, "SimplePoseGraph::loadVocabulary() 开始加载词汇表: %s", voc_path_.c_str());
  voc_ = std::make_shared<BriefVocabulary>(voc_path_);
  db_ = std::make_shared<BriefDatabase>();
  db_->setVocabulary(*voc_, false, 0);
  
  droslog(LogLevel::INFO, "SimplePoseGraph::loadVocabulary() 词汇表加载完成");
}

void SimplePoseGraph::clearDatabase()
{
  // 只清空数据库，保留词汇表
  if (db_) {
    db_->clear();
    droslog(LogLevel::INFO, "SimplePoseGraph::clearDatabase() 数据库已清空，词汇表保留");
  }
  
  // 2026-01-11: 清空 EntryId 到 KeyFrame index 的映射
  {
    std::lock_guard<std::mutex> lock(entry_map_mutex_);
    entry_to_kf_index_.clear();
  }
  
  // 清空关键帧列表
  {
    std::lock_guard<std::mutex> lock(keyframelist_mutex_);
    keyframelist_.clear();
  }
  global_index = 0;
}

void SimplePoseGraph::addKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe) 
{
  db_->add(keyframe->brief_descriptors);
  
  // 记录 DBoW2 EntryId 到 KeyFrame index 的映射
  // DBoW2 的 EntryId 是按添加顺序递增的（0, 1, 2, ...）
  // 但 KeyFrame index 可能不连续
  {
    std::lock_guard<std::mutex> lock(entry_map_mutex_);
    entry_to_kf_index_.push_back(keyframe->index);
  }
}

void SimplePoseGraph::addKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop) 
{
  if (global_index >= 1000) {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::WARN, "SimplePoseGraph::addKeyFrame() 关键帧数目过多, 停止关键帧添加");
    }
    return;
  }
  cur_kf->index = global_index++;
  addKeyFrameIntoVoc(cur_kf);

  {
    static SimpleLogFilter log_filter(5000);
    if (log_filter.Output(GetNow_Steady())) {
      droslog(LogLevel::INFO, "SimplePoseGraph::addKeyFrame() 关键帧数目: %d", global_index);
    }
  }

  {
    std::lock_guard<std::mutex> lock(keyframelist_mutex_);
    keyframelist_.push_back(cur_kf);
  }

  int loop_index = -1;  
  if (flag_detect_loop)
  {
    loop_index = detectLoop(cur_kf, cur_kf->index);  
  }    
  if (loop_index >= 0) {
    auto loop_kf = getKeyFrame(loop_index);
    if (cur_kf->findConnection(loop_kf.get())) {
      droslog(LogLevel::INFO, "SimplePoseGraph::addKeyFrame() 检测到回环: %d -> %d", cur_kf->index, loop_kf->index);
    }
  }
}

void SimplePoseGraph::loadKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop) 
{
  // 2026-01-11: 修复 Bug - 不再覆盖关键帧的原始 index
  // 关键帧的 index 在构造时已经从文件中读取，必须保持一致
  // 否则会导致 DBoW2 的 entry_to_kf_index_ 与 SpatialMapManager 的 index_to_keyframe_ 不一致
  
  // 更新 global_index 为当前最大值 + 1（用于后续新帧）
  if (cur_kf->index >= global_index) {
    global_index = cur_kf->index + 1;
  }
  
  addKeyFrameIntoVoc(cur_kf);

  // 2026-01-11: 不再保存到 keyframelist_，统一由 SpatialMapManager 管理
  // 关键帧会在 SpatialMapManager::loadFromDirectory() 中加载并注册到 index_to_keyframe_
  // keyframelist_ 只在建图模式下使用（兼容 loopCorrection）
  if (!spatial_manager_) {
    // 建图模式或未设置 spatial_manager_，保持原有行为
    std::lock_guard<std::mutex> lock(keyframelist_mutex_);
    keyframelist_.push_back(cur_kf);
  }
}

std::shared_ptr<KeyFrame> SimplePoseGraph::getKeyFrame(int index)  
{
  // 2026-01-11: 优先从 SpatialMapManager 获取（支持冷数据按需加载）
  if (spatial_manager_) {
    auto kf = spatial_manager_->getKeyFrameByIndex(index);
    if (kf) {
      return kf;
    }
  }
  
  // 回退：从 keyframelist_ 查找（兼容建图模式）
  std::lock_guard<std::mutex> lock(keyframelist_mutex_);
  for (auto& kf : keyframelist_)
  {
    if (kf && kf->index == index)
      return kf;
  }
  return nullptr;
}

std::vector<std::shared_ptr<KeyFrame>> SimplePoseGraph::getAllKeyFrames() {
  // 2026-01-11: 优先从 SpatialMapManager 获取
  if (spatial_manager_) {
    return spatial_manager_->getAllIndexedKeyFrames();
  }
  
  // 回退：从 keyframelist_ 获取（兼容建图模式）
  std::lock_guard<std::mutex> lock(keyframelist_mutex_);
  std::vector<std::shared_ptr<KeyFrame>> result;
  result.reserve(keyframelist_.size());
  for (auto& kf : keyframelist_) {
    if (kf) {
      result.push_back(kf);
    }
  }
  return result;
}

int SimplePoseGraph::getKeyFrameCount() {
  // 2026-01-11: 优先从 SpatialMapManager 获取
  if (spatial_manager_) {
    return spatial_manager_->getTotalKeyFrameCount();
  }
  
  // 回退：从 keyframelist_ 获取（兼容建图模式）
  std::lock_guard<std::mutex> lock(keyframelist_mutex_);
  return static_cast<int>(keyframelist_.size());
}

int SimplePoseGraph::cleanupUnindexedKeyFrames(const std::vector<std::shared_ptr<KeyFrame>>& validKeyFrames) {
  droslog(LogLevel::INFO, "SimplePoseGraph::cleanupUnindexedKeyFrames() 开始清理未索引的关键帧");
  
  // 构建有效关键帧指针集合，用于快速查找
  std::set<KeyFrame*> valid_kf_set;
  for (const auto& kf : validKeyFrames) {
    if (kf) {
      valid_kf_set.insert(kf.get());
    }
  }
  
  int before_count = 0;
  int removed_count = 0;
  
  {
    std::lock_guard<std::mutex> lock(keyframelist_mutex_);
    before_count = static_cast<int>(keyframelist_.size());
    
    // 遍历关键帧列表，移除未被索引的帧
    auto it = keyframelist_.begin();
    while (it != keyframelist_.end()) {
      if (*it && valid_kf_set.find((*it).get()) == valid_kf_set.end()) {
        // 该关键帧未在空间索引中，需要移除
        it = keyframelist_.erase(it);
        removed_count++;
      } else {
        ++it;
      }
    }
  }
  
  // 重建词袋数据库（只包含有效帧）
  if (removed_count > 0 && db_ && voc_) {
    droslog(LogLevel::INFO, "SimplePoseGraph::cleanupUnindexedKeyFrames() 重建词袋数据库...");
    
    // 清空旧数据库
    db_->clear();
    
    // 2026-01-11: 同时清空并重建 EntryId 映射表
    {
      std::lock_guard<std::mutex> lock(entry_map_mutex_);
      entry_to_kf_index_.clear();
    }
    
    // 重新添加有效关键帧到词袋
    std::lock_guard<std::mutex> lock(keyframelist_mutex_);
    for (const auto& kf : keyframelist_) {
      if (kf) {
        db_->add(kf->brief_descriptors);
        // 2026-01-11: 同步更新映射表
        {
          std::lock_guard<std::mutex> elock(entry_map_mutex_);
          entry_to_kf_index_.push_back(kf->index);
        }
      }
    }
    
    droslog(LogLevel::INFO, "SimplePoseGraph::cleanupUnindexedKeyFrames() 词袋数据库重建完成，包含 %zu 帧", 
        keyframelist_.size());
  }
  
  droslog(LogLevel::INFO, "SimplePoseGraph::cleanupUnindexedKeyFrames() 清理完成: 原有=%d, 移除=%d, 保留=%d",
      before_count, removed_count, before_count - removed_count);
  
  return removed_count;
}

// 地图结构
// map_root_dir/vmap/meta_map.txt: 元数据, 记录所有子地图名(时间-到sec), version vmap_name
// map_root_dir/vmap/vmap_name: 子地图
// v1:
// vmap_name/feats/: 保存所有的描述子和关键点文件
// vmap_name/imgs/: 关键帧图像, 调试用
// vmap_name/pose_graph.txt: 关键帧位姿图文件
// vmap_name/as_vmap.png: 平面视觉地图, 轨迹+特征点云
// v2:
// vmap_name/as_vision.map: 包含关键帧位姿+描述子+关键点
// vmap_name/as_vmap.png: 平面视觉地图, 轨迹+特征点云
// vmap_name/imgs: 关键帧图像, 调试用
const std::string k_meta_map_fn = "meta_map.txt";  
const std::string k_vmap_img_fn = "as_vmap.png";  
void SimplePoseGraph::saveMap(std::string map_path)
{
  // 地图元信息:地图版本号, 创建时间
  droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 即将保存地图: %s", map_path.c_str());

  if (map_path.back() != '/')
    map_path += "/";

  std::string vmap_path = map_path + "vmap/";
  if (!IsDirExisting(vmap_path.c_str())) {
    CreateDir(vmap_path.c_str());
  }
  droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 将保存地图在: %s", vmap_path.c_str());
  
  const std::string new_sm_name = GetCurTimeStamp_Sec();
  const std::string vmap_version = "V1";
  
  std::map<std::string, std::string> submaps;
  // 先备份已存meta_map.txt
  std::string meta_map_fn = vmap_path + k_meta_map_fn;
  if (IsFileExisting(meta_map_fn.c_str())) {
    std::ifstream fin(meta_map_fn);
    std::string tmp_version, tmp_name;
    while (fin >> tmp_name >> tmp_version) {
      if (tmp_version.size() == 2 && tmp_name.size() == 15) {
        submaps[tmp_name] = tmp_version;
        droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 备份已存子地图信息: %s, version=%s", tmp_name.c_str(), tmp_version.c_str());
      } else {
        droslog(LogLevel::WARN, "SimplePoseGraph::saveMap(): 读到错误子地图信息: %s, version=%s", tmp_name.c_str(), tmp_version.c_str());
        break;
      }
    }
    fin.close();
  }

  std::string submap_path = vmap_path + new_sm_name + "/";  
  CreateDir(submap_path.c_str());
  std::string feats_path = submap_path + "feats/";
  CreateDir(feats_path.c_str());
  std::string submap_img_path = submap_path + "imgs/";
  if (DEBUG_IMAGE) {
    CreateDir(submap_img_path.c_str());
  }
  droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 保存新的的子地图, %s, 是否保存图片: %d", new_sm_name.c_str(), DEBUG_IMAGE);

  std::string pg_fp = submap_path + "pose_graph.txt";
  FILE *pFile = fopen (pg_fp.c_str(),"w");

  std::vector<MapDrawer::TimedPose> org_traj, aligned_traj;

  MapDrawer::TimedPose tmp_org_pose, tmp_aligned_pose;
  float min_px = 0.f, max_px = 0.f;
  float min_py = 0.f, max_py = 0.f;

  {
    std::lock_guard<std::mutex> lock(keyframelist_mutex_);  
    std::list<std::shared_ptr<KeyFrame>>::iterator it;
    for (it = keyframelist_.begin(); it != keyframelist_.end(); it++)
    {
      std::string image_path, brief_path, keypoints_path;
      if (DEBUG_IMAGE)
      {
        image_path = submap_img_path + std::to_string((*it)->index) + ".png";
        cv::imwrite(image_path.c_str(), (*it)->image);
      }
      Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
      Quaterniond PG_tmp_Q{(*it)->R_w_i};
      Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
      Vector3d PG_tmp_T = (*it)->T_w_i;
      RefLocInfo rli = (*it)->ref_loc_info_;

      tmp_org_pose.timestamp = (*it)->time_stamp;
      tmp_org_pose.xyz[0] = VIO_tmp_T[0];
      tmp_org_pose.xyz[1] = VIO_tmp_T[1];
      tmp_org_pose.xyz[2] = VIO_tmp_T[2];

      min_px = std::min(min_px, tmp_org_pose.xyz[0]);
      max_px = std::max(max_px, tmp_org_pose.xyz[0]);
      min_py = std::min(min_py, tmp_org_pose.xyz[1]);
      max_py = std::max(max_py, tmp_org_pose.xyz[1]);

      {
        auto rpy = GetEulerRPY(VIO_tmp_Q);
        tmp_org_pose.rpy[0] = rpy[0];
        tmp_org_pose.rpy[1] = rpy[1];
        tmp_org_pose.rpy[2] = rpy[2];
      }
      org_traj.push_back(tmp_org_pose);

      tmp_aligned_pose.timestamp = (*it)->time_stamp;
      tmp_aligned_pose.xyz[0] = PG_tmp_T[0];
      tmp_aligned_pose.xyz[1] = PG_tmp_T[1];
      tmp_aligned_pose.xyz[2] = PG_tmp_T[2];

      min_px = std::min(min_px, tmp_aligned_pose.xyz[0]);
      max_px = std::max(max_px, tmp_aligned_pose.xyz[0]);
      min_py = std::min(min_py, tmp_aligned_pose.xyz[1]);
      max_py = std::max(max_py, tmp_aligned_pose.xyz[1]);

      {
        auto rpy = GetEulerRPY(PG_tmp_Q);
        tmp_aligned_pose.rpy[0] = rpy[0];
        tmp_aligned_pose.rpy[1] = rpy[1];
        tmp_aligned_pose.rpy[2] = rpy[2];
      }
      aligned_traj.push_back(tmp_aligned_pose);
  
      fprintf (pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d %f %d %f %f %f %f %f %f %f %f %f %f %f %f\n",(*it)->index, (*it)->time_stamp, 
                                VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(), 
                                PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(), 
                                VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(), 
                                PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(), 
                                (*it)->loop_index, 
                                (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
                                (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
                                (int)(*it)->keypoints.size(), rli.timestamp, rli.type, rli.xyz[0], rli.xyz[1], rli.xyz[2],
                                rli.cov(0,0), rli.cov(0,1), rli.cov(0,2), rli.cov(1,0), rli.cov(1,1), rli.cov(1,2), rli.cov(2,0), rli.cov(2,1), rli.cov(2,2));
  
      // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
      if ((*it)->keypoints.size() == (*it)->brief_descriptors.size()) {  
        brief_path = feats_path + std::to_string((*it)->index) + "_briefdes.dat";
        std::ofstream brief_file(brief_path, std::ios::binary);
        keypoints_path = feats_path + std::to_string((*it)->index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "w");
        for (int i = 0; i < (int)(*it)->keypoints.size(); i++)
        {
          brief_file << (*it)->brief_descriptors[i] << endl;
          fprintf(keypoints_file, "%f %f %f %f\n", (*it)->keypoints[i].pt.x, (*it)->keypoints[i].pt.y, 
                                                    (*it)->keypoints_norm[i].pt.x, (*it)->keypoints_norm[i].pt.y);
        }
        brief_file.close();
        fclose(keypoints_file);
      } else {
        droslog(LogLevel::ERROR, "SimplePoseGraph::saveMap(): keypoints.size() != brief_descriptors.size(), KFid=%d", (*it)->index);
      }
    }
    fclose(pFile);
  }

  // 保存地图图片
  {
    min_px = std::max(min_px - 2.0, -200.0);
    max_px = std::min(max_px + 2.0, 200.0);
    min_py = std::max(min_py - 2.0, -200.0);
    max_py = std::min(max_py + 2.0, 200.0);

    MapDrawer::CanvasParams canvas_params;
    canvas_params.resolution = 0.05;
    canvas_params.height = (max_px - min_px) / canvas_params.resolution;
    canvas_params.width  = (max_py - min_py) / canvas_params.resolution;
    canvas_params.org_xy[0] = max_py / canvas_params.resolution;
    canvas_params.org_xy[1] = max_px  / canvas_params.resolution;

    MapDrawer drawer;
    drawer.InitCanvas(canvas_params);

    drawer.DrawGrid();
    drawer.DrawOrgP();
    MapDrawer::TrajConfig traj_config;
    traj_config.cc_bar_type = 0;
    traj_config.color = cv::Scalar(100, 100, 100);
    drawer.DrawTraj(org_traj, traj_config);
    traj_config.cc_bar_type = 1;
    drawer.DrawTraj(aligned_traj, traj_config);
    auto map_img = drawer.GetMap();
    std::string map_img_fn = submap_path + "map_traj.png";
    cv::imwrite(map_img_fn, map_img);
  }

  // 更新元地图信息
  submaps[new_sm_name] = vmap_version;
  {
    droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 更新地图元数据...");
    std::ofstream fout(meta_map_fn);
    for (const auto &it : submaps) {
      fout << it.first << " " << it.second << std::endl;
      droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 写入子地图信息: name=%s, version=%s", it.first.c_str(), it.second.c_str());
    }
    fout.close();
    droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 更新地图元数据 done");
  }
  
  // 2026-01-11: 保存 DBoW2 数据库（用于后续对比分析）
  // 这是建图时的原始词袋，作业时会动态更新内存中的副本
  if (db_) {
    std::string dbow_db_path = submap_path + "brief_db_original.yml";
    try {
      db_->save(dbow_db_path);
      droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): 保存 DBoW2 数据库: %s, 包含 %d 条目", 
              dbow_db_path.c_str(), (int)db_->size());
    } catch (const std::exception& e) {
      droslog(LogLevel::WARN, "SimplePoseGraph::saveMap(): 保存 DBoW2 数据库失败: %s", e.what());
    }
  }

  droslog(LogLevel::INFO, "SimplePoseGraph::saveMap(): save Map done: %s", submap_path.c_str());
}

int SimplePoseGraph::loadMap(std::string map_path)
{
  droslog(LogLevel::INFO, "SimplePoseGraph::loadMap(): 即将加载地图: %s", map_path.c_str());

  if (map_path.back() != '/')
    map_path += "/";

  std::string vmap_path = map_path + "vmap/";
  if (!IsDirExisting(vmap_path.c_str())) {
    droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap(): 不存在视觉地图: %s", vmap_path.c_str());
    return 0;
  }
  
  // 读取所有的子地图
  std::map<std::string, std::string> submaps;
  std::string meta_map_fn = vmap_path + k_meta_map_fn;
  if (IsFileExisting(meta_map_fn.c_str())) {
    std::ifstream fin(meta_map_fn);
    std::string tmp_name, tmp_version;
    while (fin >> tmp_name >> tmp_version) {
      if (tmp_name.size() == 15 && tmp_version.size() == 2) {
        submaps[tmp_name] = tmp_version;
        droslog(LogLevel::INFO, "SimplePoseGraph::loadMap(): 读到子地图信息: name=%s, version=%s", tmp_name.c_str(), tmp_version.c_str());
      } else {
        droslog(LogLevel::WARN, "SimplePoseGraph::loadMap(): 读到错误子地图信息: name=%s, version=%s", tmp_name.c_str(), tmp_version.c_str());
        break;
      }
    }
    fin.close();
  } else {
    droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap(): 视觉地图子地图名文件不存在: %s", meta_map_fn.c_str());
    return 0;
  }

  if (submaps.size() == 0) {
    droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap(): 有效子地图数目为空!");
    return 0;
  }

  for (const auto &it : submaps) {
    std::string sm_name = it.first;
    std::string sm_ver = it.second;

    droslog(LogLevel::INFO, "SimplePoseGraph::loadMap(): 加载子地图: name=%s, version=%s, 是否加载图片: %d", sm_name.c_str(), sm_ver.c_str(), DEBUG_IMAGE);
    std::string submap_path = vmap_path + sm_name + "/";

    std::string submap_img_path = submap_path + "imgs/";
    std::string feats_path = submap_path + "feats/";
    std::string pg_fp = submap_path + "pose_graph.txt";

    // 检查文件
    if (!IsDirExisting(feats_path.c_str())) {
      droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap(): 加载子地图出错: name=%s, 没有 feats文件夹", sm_name.c_str());
      return 0;
    }
    if (!IsFileExisting(pg_fp.c_str())) {
      droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap(): 加载子地图出错: name=%s, 没有 pose_graph.txt", sm_name.c_str());
      return 0;
    }

    FILE * pFile = fopen (pg_fp.c_str(),"r");

    int index;
    double time_stamp;
    double VIO_Tx, VIO_Ty, VIO_Tz;
    double PG_Tx, PG_Ty, PG_Tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    int keypoints_num;
    // Eigen::Matrix<double, 8, 1 > loop_info;
    double rli_ts;
    int rli_type;
    double rli_x, rli_y, rli_z;
    double rli_cov00, rli_cov01, rli_cov02, rli_cov10, rli_cov11, rli_cov12, rli_cov20, rli_cov21, rli_cov22;

    int cnt = 0;
    while (fscanf(pFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &index, &time_stamp, 
                                  &VIO_Tx, &VIO_Ty, &VIO_Tz, 
                                  &PG_Tx, &PG_Ty, &PG_Tz, 
                                  &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, 
                                  &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, 
                                  &loop_index,
                                  &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3, 
                                  &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                                  &keypoints_num, &rli_ts, &rli_type, &rli_x, 
                                  &rli_y, &rli_z, &rli_cov00, &rli_cov01, &rli_cov02, &rli_cov10, &rli_cov11, &rli_cov12, &rli_cov20, &rli_cov21, &rli_cov22) != EOF) 
    {
      cv::Mat image;
      std::string image_path, descriptor_path;
      if (DEBUG_IMAGE) {
        image_path = submap_img_path + to_string(index) + ".png";
        if (IsFileExisting(image_path.c_str())) {
          image = cv::imread(image_path.c_str(), 0);
        }
      }

      Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
      Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
      Quaterniond VIO_Q;
      VIO_Q.w() = VIO_Qw;
      VIO_Q.x() = VIO_Qx;
      VIO_Q.y() = VIO_Qy;
      VIO_Q.z() = VIO_Qz;
      Quaterniond PG_Q;
      PG_Q.w() = PG_Qw;
      PG_Q.x() = PG_Qx;
      PG_Q.y() = PG_Qy;
      PG_Q.z() = PG_Qz;
      Matrix3d VIO_R, PG_R;
      VIO_R = VIO_Q.toRotationMatrix();
      PG_R = PG_Q.toRotationMatrix();
      Eigen::Matrix<double, 8, 1 > loop_info;
      loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

      RefLocInfo rli;
      rli.timestamp = rli_ts;
      rli.type = rli_type;
      rli.xyz << rli_x, rli_y, rli_z;
      rli.cov << rli_cov00, rli_cov01, rli_cov02, rli_cov10, rli_cov11, rli_cov12, rli_cov20, rli_cov21, rli_cov22;

      // load keypoints, brief_descriptors   
      string brief_path = feats_path + std::to_string(index) + "_briefdes.dat";
      std::string keypoints_path = feats_path + std::to_string(index) + "_keypoints.txt";
      if (!IsFileExisting(brief_path.c_str())) {
        droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap() brief 文件不存在: %s", brief_path.c_str());
        return 0;
      }
      if (!IsFileExisting(keypoints_path.c_str())) {
        droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap() keypoints 文件不存在: %s", keypoints_path.c_str());
        return 0;
      }

      std::ifstream brief_file(brief_path, std::ios::binary);
      FILE *keypoints_file;
      keypoints_file = fopen(keypoints_path.c_str(), "r");
      std::vector<cv::KeyPoint> keypoints;
      std::vector<cv::KeyPoint> keypoints_norm;
      std::vector<BRIEF::bitset> brief_descriptors;
      for (int i = 0; i < keypoints_num; i++)
      {
        BRIEF::bitset tmp_des;
        brief_file >> tmp_des;
        brief_descriptors.push_back(tmp_des);
        cv::KeyPoint tmp_keypoint;
        cv::KeyPoint tmp_keypoint_norm;
        double p_x, p_y, p_x_norm, p_y_norm;
        if(!fscanf(keypoints_file,"%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
        {
          droslog(LogLevel::ERROR, "SimplePoseGraph::loadMap(): fail to load keypoints, i=%d", i);
        }
        tmp_keypoint.pt.x = p_x;
        tmp_keypoint.pt.y = p_y;
        tmp_keypoint_norm.pt.x = p_x_norm;
        tmp_keypoint_norm.pt.y = p_y_norm;
        keypoints.push_back(tmp_keypoint);
        keypoints_norm.push_back(tmp_keypoint_norm);
      }
      brief_file.close();
      fclose(keypoints_file);

      std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors);
      keyframe->SetRefLocInfo(rli);
      loadKeyFrame(keyframe, 0);
      cnt++;
    }
    fclose (pFile);
    droslog(LogLevel::INFO, "SimplePoseGraph::loadMap(): 加载子地图: name=%s, version=%s, 加载了 %d keyframes", sm_name.c_str(), sm_ver.c_str(), cnt);
    
    // 2026-01-11: 检查是否存在保存的 DBoW2 数据库文件（用于对比分析）
    std::string dbow_db_path = submap_path + "brief_db_original.yml";
    if (IsFileExisting(dbow_db_path.c_str())) {
      droslog(LogLevel::INFO, "SimplePoseGraph::loadMap(): 发现原始 DBoW2 数据库文件: %s", dbow_db_path.c_str());
      // 注意：这里不加载它，因为我们已经通过 loadKeyFrame() 重建了词袋
      // 原始文件保留用于后续对比分析（如检测地图变化、评估重定位质量等）
    }
  }
  
  // 记录加载完成后的 DBoW2 数据库大小
  if (db_) {
    droslog(LogLevel::INFO, "SimplePoseGraph::loadMap(): DBoW2 数据库加载完成，大小: %d 条目", (int)db_->size());
  }
  
  return 1;
}

int SimplePoseGraph::detectLoop(const std::shared_ptr<KeyFrame> &cur_kf, int frame_index)  
{
  bool found = false;
  // 1. detect loop
  DBoW2::QueryResults ret;
  
  // 使用 DBoW2 EntryId 作为 max_id 参数
  // 需要将 frame_index 转换为对应的 EntryId
  int max_entry_id = -1;
  {
    std::lock_guard<std::mutex> lock(entry_map_mutex_);
    // 查找当前帧对应的 EntryId（应该是最后一个）
    max_entry_id = static_cast<int>(entry_to_kf_index_.size()) - 1;
    // 排除最近 25 个条目，避免匹配到相邻帧
    max_entry_id = std::max(-1, max_entry_id - 25);
  }
  
  db_->query(cur_kf->brief_descriptors, ret, 4, max_entry_id);

  // a good match with its neigbhours
  if (ret.size() >= 1 && ret[0].Score > 0.05) {
    for (unsigned int i = 1; i < ret.size(); i++)
    {
      if (ret[i].Score > 0.015)
      {
        found = true;
        break;  // 1118 优化：找到一个满足条件的就退出
      }
    }
  }

  int result_kf_index = -1;
  if (found && max_entry_id > 0) 
  {
    // 2026-01-11: 找到最佳匹配的 EntryId，然后转换为 KeyFrame index
    int best_entry_id = -1;
    for (unsigned int i = 0; i < ret.size(); i++)
    {
      if (best_entry_id == -1 || (static_cast<int>(ret[i].Id) < best_entry_id && ret[i].Score > 0.015))
      {
        best_entry_id = ret[i].Id;
      }
    }
    
    // 将 DBoW2 EntryId 转换为 KeyFrame index
    if (best_entry_id >= 0) {
      std::lock_guard<std::mutex> lock(entry_map_mutex_);
      if (best_entry_id < static_cast<int>(entry_to_kf_index_.size())) {
        result_kf_index = entry_to_kf_index_[best_entry_id];
      }
    }
  }
  return result_kf_index;
}

int SimplePoseGraph::relocalization(const std::shared_ptr<KeyFrame> &cur_kf, Eigen::Vector3d &pos, Eigen::Quaterniond &quat, int type, bool is_first_reloc)
{
  int reloc_idx = -1;
  bool found = false;
  
  // 1. DBoW2 查询候选帧
  DBoW2::QueryResults ret;
  db_->query(cur_kf->brief_descriptors, ret, 10, -1);  // 2026-01-11: 增加到10个候选

  // a good match with its neigbhours
  if (ret.size() >= 1 && ret[0].Score > 0.025) {
    for (unsigned int i = 1; i < ret.size(); i++)
    {
      if (ret[i].Score > 0.015)
      {
        found = true;
        break;  // 找到一个满足条件的邻居就退出
      }
    }
  }

  if (!found) {
    // 2026-01-12: DBoW2 匹配失败统计（降频日志）
    static int dbow_fail_cnt = 0;
    static SimpleLogFilter dbow_fail_filter(10000);  // 10秒一次
    dbow_fail_cnt++;
    if (dbow_fail_filter.Output(GetNow_Steady())) {
      float top_score = ret.size() > 0 ? ret[0].Score : 0.0f;
      droslog(LogLevel::INFO, "relocalization(): DBoW2 匹配失败统计: 累计%d次, top_score=%.3f, candidates=%d", 
              dbow_fail_cnt, top_score, (int)ret.size());
    }
    return reloc_idx;
  }

  int cur_id = cur_kf->index;
  
  // 2026-01-11: 两轮匹配策略
  // 第一轮：只尝试热数据（避免磁盘 I/O）
  // 第二轮：尝试冷数据（按需加载）
  
  // 2026-01-11: 将 DBoW2 EntryId 转换为 KeyFrame index
  // 按得分排序候选帧（存储的是 KeyFrame index，不是 EntryId）
  std::vector<std::pair<int, float>> candidates;
  {
    std::lock_guard<std::mutex> lock(entry_map_mutex_);
    for (unsigned int i = 0; i < ret.size(); i++) {
      int entry_id = ret[i].Id;
      // 将 EntryId 转换为 KeyFrame index
      if (entry_id < 0 || entry_id >= static_cast<int>(entry_to_kf_index_.size())) {
        continue;
      }
      int kf_index = entry_to_kf_index_[entry_id];
      
      if (ret[i].Score > 0.015 && (0 == type || (1 == type && cur_id > kf_index + 25))) {
        candidates.push_back({kf_index, ret[i].Score});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });
  
  // 2026-01-12: DBoW2 匹配成功统计（降频日志）
  static int dbow_success_cnt = 0;
  static SimpleLogFilter dbow_success_filter(10000);  // 10秒一次
  dbow_success_cnt++;
  if (dbow_success_filter.Output(GetNow_Steady())) {
    float top_score = ret[0].Score;
    droslog(LogLevel::INFO, "relocalization(): DBoW2 匹配成功统计: 累计%d次, top_score=%.3f, candidates=%d", 
            dbow_success_cnt, top_score, (int)candidates.size());
  }
  
  // 第一轮：优先尝试热数据
  for (const auto& cand : candidates) {
    int cand_index = cand.first;  // 这里已经是 KeyFrame index
    float cand_score = cand.second;
    
    // 检查是否是热数据
    if (spatial_manager_ && !spatial_manager_->isHotData(cand_index)) {
      continue;  // 跳过冷数据，第二轮再处理
    }
    
    std::shared_ptr<KeyFrame> loop_kf = getKeyFrame(cand_index);
    if (!loop_kf) continue;
    
    reloc_idx = tryRelocWithCandidate(cur_kf, loop_kf, cand_score, pos, quat, type, is_first_reloc);
    if (reloc_idx >= 0) {
      return reloc_idx;
    }
  }
  
  // 第二轮：尝试冷数据（按需加载）
  for (const auto& cand : candidates) {
    int cand_index = cand.first;  // 这里已经是 KeyFrame index
    float cand_score = cand.second;
    
    // 第一轮已经尝试过热数据
    if (spatial_manager_ && spatial_manager_->isHotData(cand_index)) {
      continue;
    }
    
    // 冷数据，getKeyFrame 会自动从磁盘加载
    std::shared_ptr<KeyFrame> loop_kf = getKeyFrame(cand_index);
    if (!loop_kf) continue;
    
    reloc_idx = tryRelocWithCandidate(cur_kf, loop_kf, cand_score, pos, quat, type, is_first_reloc);
    if (reloc_idx >= 0) {
      return reloc_idx;
    }
  }
  
  return reloc_idx;
}

// 2026-01-11: 辅助函数，尝试与候选帧进行重定位
// 2026-01-13: 参考 VioTracker 的思路，首次重定位不依赖 relative_t 验证
//             而是直接用 PnP 结果与 GPS 比较，因为 relative_t 依赖 VIO 坐标系对齐
int SimplePoseGraph::tryRelocWithCandidate(
    const std::shared_ptr<KeyFrame>& cur_kf,
    const std::shared_ptr<KeyFrame>& loop_kf,
    float score,
    Eigen::Vector3d& pos,
    Eigen::Quaterniond& quat,
    int type,
    bool is_first_reloc)
{
  if (!loop_kf || loop_kf->brief_descriptors.empty()) {
    return -1;
  }
  
  int cur_id = cur_kf->index;
  int loop_id = loop_kf->index;
  
  // 2026-01-13: 传入 is_first_reloc 参数
  // 参考 VioTracker 的思路：
  // 1. VioTracker 使用多帧累积 + spa_align 优化来计算 VIO→地图 变换
  // 2. 首次对齐时需要足够多的约束（valid_cc_cnt > size/2）
  // 3. 这里首次重定位时跳过 relative_t 验证，只依赖 GPS 验证
  
  if (cur_kf->findConnection(loop_kf.get(), is_first_reloc)) {
    // 2026-01-11: 修复 - 使用 getPose() 获取优化后的位姿，而非 getVioPose()
    Eigen::Vector3d w_P_loop;
    Eigen::Matrix3d w_R_loop;
    loop_kf->getPose(w_P_loop, w_R_loop);

    Eigen::Vector3d relative_t = cur_kf->getLoopRelativeT();
    Eigen::Quaterniond relative_q = cur_kf->getLoopRelativeQ();
    
    Eigen::Vector3d w_P_cur = w_P_loop + w_R_loop * relative_t;
    Eigen::Matrix3d w_R_cur = w_R_loop * relative_q.toRotationMatrix();
    
    // 二次验证：双重保险，防止异常情况
    // 2026-01-13: 完全参考 VioTracker，首次重定位不做验证
    double ddist = relative_t.norm();
    double relative_yaw = cur_kf->getLoopRelativeYaw();
    
    if (is_first_reloc) {
      // 首次重定位：不做二次验证
      // 完全参考 VioTracker：让 PnP 结果直接 feed 到 VioTracker
      // 由 spa_align 图优化来判断是否可信
      // findConnection() 已经做了基本的 roll/pitch 异常检测
    } else {
      // 已对齐后：完整验证
      if (ddist > 0.5 || std::abs(relative_yaw) > 0.3) {
        // 理论上不应该进入这里，因为 findConnection() 已经验证过
        // 但作为防御性编程，记录异常情况
        droslog(LogLevel::WARN, "tryRelocWithCandidate(): 二次验证失败(异常) cur=%d->loop=%d, ddist=%.2f, yaw=%.2f", 
            cur_id, loop_id, ddist, relative_yaw);
        return -1;
      }
    }
    
    pos = w_P_cur;
    quat = Eigen::Quaterniond(w_R_cur);

    droslog(LogLevel::INFO, "tryRelocWithCandidate(): 成功 cur=%d->loop=%d, score=%.3f, rel_t=(%.2f,%.2f,%.2f)", 
        cur_id, loop_id, score, relative_t.x(), relative_t.y(), relative_t.z());
    return loop_id;
  }
  
  return -1;
}

// ========== 空间索引管理器设置 - 2025-12-25 ==========
void SimplePoseGraph::setSpatialMapManager(SpatialMapManager* manager) {
  spatial_manager_ = manager;
}

// 注：relocalizationWithSpatialIndex() 已删除 - 2026-01-12
// 原因：VIO 漂移可能导致空间索引查询位置错误
// 现在统一使用 relocalization()（全局 DBoW2 搜索）+ findConnection() 验证

int SimplePoseGraph::loopCorrection() {
  droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() 开始全图对齐修正");

  std::vector<TimedAlignNode> pg_nodes;
  int rtk_ref_size = 0;
  int station_ref_size = 0;
  int loop_size = 0;
  std::lock_guard<std::mutex> lock(keyframelist_mutex_);
  std::list<std::shared_ptr<KeyFrame>>::iterator it;
  auto TF_v2g = TFHelper::Instance()->Vio2Gps_t();
  // T_v = T_b + R * TF_v2b - TF_v2b
  // T_b = T_v - R * TF_v2b + TF_v2b
  for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
    TimedAlignNode node;
    node.id = (*it)->index;
    node.timestamp = (*it)->time_stamp;
    Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
    Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
    auto pose = TFHelper::Instance()->TF_Vio2Gps(VIO_tmp_T, VIO_tmp_Q);
    node.pos = pose.pos + TF_v2g;
    node.quat = pose.quat;

    const auto& rli = (*it)->ref_loc_info_;
    if (rli.type == 0) {  // 在桩
      station_ref_size++;
      auto sp = std::make_shared<NodeRefPose>();
      sp->ref_pos_valid = true;
      sp->ref_pos = rli.xyz;
      sp->ref_pos_cov = rli.cov;

      sp->ref_quat_valid = true;
      sp->ref_quat = Quaterniond::Identity();
      sp->ref_quat_cov << 0.0001, 0.0, 0.0,
                           0.0, 0.0001, 0.0,
                           0.0, 0.0, 0.0001;

      node.ref_pose = sp;
    } else if (rli.type == 1) { // RTK
      rtk_ref_size++;
      auto sp = std::make_shared<NodeRefPose>();
      sp->ref_pos_valid = true;
      sp->ref_pos = rli.xyz;
      sp->ref_pos_cov = rli.cov;
      
      node.ref_pose = sp;
    }

    if ((*it)->has_loop) {
      loop_size++;
      auto sp = std::make_shared<NodeLoopInfo>();
      sp->ref_id = (*it)->loop_index;
      sp->relative_t = Eigen::Vector3d((*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2));
      sp->relative_q = Eigen::Quaterniond((*it)->loop_info(3), (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6));
      sp->relative_t_cov = Eigen::Matrix3d::Identity() * loop_pos_cov;
      sp->relative_q_cov = Eigen::Matrix3d::Identity() * loop_quat_cov;

      node.loop_info = sp;

      auto rpy = GetEulerRPY(sp->relative_q);
      droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() add loop-edge(cur->loop): %d -> %d, dxyz=(%.3f,%.3f,%.3f), drpy=(%.3f,%.3f,%.3f)", 
          (*it)->index, (*it)->loop_index, sp->relative_t[0], sp->relative_t[1], sp->relative_t[2], rpy[0], rpy[1], rpy[2]);
    }

    pg_nodes.push_back(node);
  }

  droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() 构建完所有节点: size=%d, rtk_ref_size=%d, station_ref_size=%d, loop_size=%d",
      (int)pg_nodes.size(), rtk_ref_size, station_ref_size, loop_size);
  

  if (rtk_ref_size <= 10 && station_ref_size <= 0 && loop_size <= 0) {
    droslog(LogLevel::ERROR, "SimplePoseGraph::loopCorrection() 参考位姿太少, 无法修正");
    return 0;
  }

  for (int i = 0; i < 5; i++) {
    auto pg = pg_nodes[i];
    droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() 对齐前: id=%d, ts=%.3f, pos=%.3f, %.3f, %.3f",
        i, pg.timestamp, pg.pos[0], pg.pos[1], pg.pos[2]);
  }
  
  auto aligned_pg = spa_align(pg_nodes);

  for (int i = 0; i < 5; i++) {
    auto pg = aligned_pg[i];
    droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() 对齐后: id=%d, ts=%.3f, pos=%.3f, %.3f, %.3f",
        i, pg.timestamp, pg.pos[0], pg.pos[1], pg.pos[2]);
  }

  if (aligned_pg.size() != pg_nodes.size()) {
    droslog(LogLevel::ERROR, "SimplePoseGraph::loopCorrection() spa_align()失败, aligned_pg.size()=%d", aligned_pg.size());
    return 0;
  } 

  droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() 开始更新所有关键帧位姿");
  int pg_idx = 0;
  for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
    auto aligned_pos = aligned_pg[pg_idx].pos;
    auto aligned_quat = aligned_pg[pg_idx].quat;

    auto aligned_vio = TFHelper::Instance()->TF_Gps2Vio(aligned_pos, aligned_quat);
    // aligned_vio.pos -= TF_v2g;
    (*it)->updatePose(aligned_vio.pos, aligned_vio.quat.toRotationMatrix());

    pg_idx++;
  }

  droslog(LogLevel::INFO, "SimplePoseGraph::loopCorrection() 全图对齐修正完成");
  return 1;
}