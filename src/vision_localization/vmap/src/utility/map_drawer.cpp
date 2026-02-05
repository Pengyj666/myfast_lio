#include "map_drawer.h"

#include <opencv2/opencv.hpp>

namespace {
  std::vector<cv::Vec3b> g_color_bar;

  const auto cv_black = cv::Scalar(0, 0, 0);
  const auto cv_white = cv::Scalar(255, 255, 255);

  const auto cv_red   = cv::Scalar(0, 0, 255);
  const auto cv_green = cv::Scalar(0, 255, 0);
  const auto cv_blue  = cv::Scalar(255, 0, 0);

  const auto cv_yellow= cv::Scalar(0, 255, 255);
  const auto cv_cyan  = cv::Scalar(255, 255, 0);
  const auto cv_magenta = cv::Scalar(255, 0, 255);

  const auto cv_gray = cv::Scalar(128, 128, 128);
  const auto cv_deep_gray = cv::Scalar(150, 150, 150);
  const auto cv_light_gray = cv::Scalar(220, 220, 220);
} // namespace 

MapDrawer::MapDrawer() : is_init_(false) {
  if (g_color_bar.empty()) {
    {
      const int Len = 400;
      int colorR=0, colorG=0, colorB=0;
      for (int i=0; i<Len; ++i) {
          colorR = std::max(0, std::min(255, 255 - 255*i*2/Len));
          colorG = std::max(0, std::min(255, 255 - 255*std::abs(Len/2-i)*2/Len));
          colorB = std::max(0, std::min(255, 255 - 255*(Len-i)*2/Len));
          g_color_bar.push_back(cv::Vec3b(colorB, colorG, colorR));
      }
    }
  }
  canvas_ = cv::Mat(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));  
  cv::putText(canvas_, "Canvas NOT INITED", cv::Point2i(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv_black, 2);
  cur_frame_ = canvas_.clone();
}

void MapDrawer::InitCanvas(const CanvasParams& canvas_params) {
  canvas_params_ = canvas_params;
  affine_mat_[0][0] = 0.0;
  affine_mat_[0][1] = -1.0/canvas_params_.resolution;
  affine_mat_[0][2] = canvas_params_.org_xy[0];
  affine_mat_[1][0] = -1.0/canvas_params_.resolution;
  affine_mat_[1][1] = 0.0;
  affine_mat_[1][2] = canvas_params_.org_xy[1];

  canvas_ = cv::Mat(canvas_params_.height, canvas_params_.width, CV_8UC3, cv::Scalar(255, 255, 255));
  cur_frame_ = canvas_.clone();
  is_init_.store(true);
}

void MapDrawer::DrawTraj(const std::vector<TimedPose>& traj, const TrajConfig& config) {
  if (!is_init_ || traj.size() < 2) return;

  double ddts = traj.back().timestamp - traj[0].timestamp;
  if (ddts < 1e-3) {
    ddts = 1.0;
  }

  for (size_t i = 0; i < traj.size(); ++i) {
    cv::Scalar cc = config.color;
    if (config.cc_bar_type == 1) {
      int idx = g_color_bar.size() * (traj[i].timestamp - traj[0].timestamp) / ddts;
      cc = g_color_bar[idx];      
    }
    float len = 8.0f;
    auto p0 = Tf2Img_f(traj[i].xyz[0], traj[i].xyz[1]);
    auto p1 = p0 - cv::Point2f(std::sin(traj[i].rpy[2]) * len, 
                               std::cos(traj[i].rpy[2]) * len);
    
    cv::arrowedLine(cur_frame_, p0, p1, cc, 1, 8, 0, 0.5);
  }
}

void MapDrawer::DrawGrid() {
  if (!is_init_) return;

  int scale = - affine_mat_[0][1];
  for (int i = scale; i < cur_frame_.rows; i += scale) {
    auto color = (i % (scale * 10) == 0 ? cv_red : cv_gray);
    cv::line(cur_frame_, cv::Point2i(0,i), cv::Point2i(cur_frame_.cols,i), color, 1);
  }
  for (int i = scale; i < cur_frame_.cols; i += scale) {
    auto color = (i % (scale * 10) == 0 ? cv_red : cv_gray);
    cv::line(cur_frame_, cv::Point2i(i,0), cv::Point2i(i,cur_frame_.rows), color, 1);
  }
}

void MapDrawer::DrawOrgP() {
  auto p0 = Tf2Img_f(0, 0);
  auto px = Tf2Img_f(1, 0);
  auto py = Tf2Img_f(0, 1);
  cv::circle(cur_frame_, p0, 3, cv_red, -1);
  cv::line(cur_frame_, p0, px, cv_red, 2);
  cv::line(cur_frame_, p0, py, cv_red, 2);
}

cv::Point2f MapDrawer::Tf2Img_f(float x, float y) {
  float u = affine_mat_[0][0] * x + affine_mat_[0][1] * y + affine_mat_[0][2];
  float v = affine_mat_[1][0] * x + affine_mat_[1][1] * y + affine_mat_[1][2];
  return cv::Point2f(u, v);
}

cv::Point2i MapDrawer::Tf2Img_i(float x, float y) {
  int u = int(affine_mat_[0][0] * x + affine_mat_[0][1] * y + affine_mat_[0][2]);
  int v = int(affine_mat_[1][0] * x + affine_mat_[1][1] * y + affine_mat_[1][2]);
  return cv::Point2i(u, v);
}

// ========== 空间索引可视化实现 ==========

cv::Scalar MapDrawer::GetDirectionColor(int direction_slot) {
  // 6 种颜色对应 6 个方向槽位
  static const cv::Scalar colors[6] = {
    cv::Scalar(0, 0, 255),     // 0: 红色 (0°-60°)
    cv::Scalar(0, 165, 255),   // 1: 橙色 (60°-120°)
    cv::Scalar(0, 255, 255),   // 2: 黄色 (120°-180°)
    cv::Scalar(0, 255, 0),     // 3: 绿色 (180°-240°)
    cv::Scalar(255, 0, 0),     // 4: 蓝色 (240°-300°)
    cv::Scalar(255, 0, 255)    // 5: 紫色 (300°-360°)
  };
  return colors[direction_slot % 6];
}

void MapDrawer::DrawKeyFrames(const std::vector<KeyFrameVis>& keyframes) {
  if (!is_init_ || keyframes.empty()) return;
  
  for (const auto& kf : keyframes) {
    auto p0 = Tf2Img_f(kf.x, kf.y);
    
    // 根据方向槽位选择颜色
    cv::Scalar color = GetDirectionColor(kf.direction_slot);
    
    // 绘制方向箭头
    float arrow_len = 6.0f;
    auto p1 = p0 - cv::Point2f(std::sin(kf.yaw) * arrow_len, 
                               std::cos(kf.yaw) * arrow_len);
    cv::arrowedLine(cur_frame_, p0, p1, color, 1, 8, 0, 0.4);
    
    // 如果有RTK约束，绘制小圆点
    if (kf.has_rtk) {
      cv::circle(cur_frame_, p0, 2, cv_green, -1);
    }
    
    // 如果有回环，绘制小方块
    if (kf.has_loop) {
      cv::rectangle(cur_frame_, 
                    cv::Point2i(p0.x - 2, p0.y - 2),
                    cv::Point2i(p0.x + 2, p0.y + 2),
                    cv_cyan, -1);
    }
  }
}

void MapDrawer::DrawCellGrid(float cell_size) {
  if (!is_init_) return;
  
  // 计算像素间隔
  float pixels_per_cell = cell_size / canvas_params_.resolution;
  
  // 绘制浅灰色 Cell 网格线
  for (float y = -100; y < 100; y += cell_size) {
    auto p0 = Tf2Img_f(-100, y);
    auto p1 = Tf2Img_f(100, y);
    cv::line(cur_frame_, p0, p1, cv_light_gray, 1);
  }
  for (float x = -100; x < 100; x += cell_size) {
    auto p0 = Tf2Img_f(x, -100);
    auto p1 = Tf2Img_f(x, 100);
    cv::line(cur_frame_, p0, p1, cv_light_gray, 1);
  }
}

void MapDrawer::DrawSubMapGrid(float submap_size) {
  if (!is_init_) return;
  
  // 绘制深灰色 SubMap 边界线
  for (float y = -100; y < 100; y += submap_size) {
    auto p0 = Tf2Img_f(-100, y);
    auto p1 = Tf2Img_f(100, y);
    cv::line(cur_frame_, p0, p1, cv_deep_gray, 2);
  }
  for (float x = -100; x < 100; x += submap_size) {
    auto p0 = Tf2Img_f(x, -100);
    auto p1 = Tf2Img_f(x, 100);
    cv::line(cur_frame_, p0, p1, cv_deep_gray, 2);
  }
}

void MapDrawer::DrawCellHeatmap(const std::vector<KeyFrameVis>& keyframes, float cell_size) {
  if (!is_init_ || keyframes.empty()) return;
  
  // 统计每个 Cell 的关键帧数量
  std::map<std::pair<int,int>, int> cell_counts;
  int max_count = 0;
  
  for (const auto& kf : keyframes) {
    auto key = std::make_pair(kf.cell_x, kf.cell_y);
    cell_counts[key]++;
    max_count = std::max(max_count, cell_counts[key]);
  }
  
  if (max_count == 0) return;
  
  // 绘制热力图
  for (const auto& pair : cell_counts) {
    int cx = pair.first.first;
    int cy = pair.first.second;
    int count = pair.second;
    
    // 计算 Cell 中心位置
    float x = (cx + 0.5f) * cell_size;
    float y = (cy + 0.5f) * cell_size;
    
    // 根据数量计算颜色（1帧浅蓝，6帧深红）
    float ratio = float(count) / 6.0f;  // 最大6个方向
    ratio = std::min(1.0f, ratio);
    
    // 从蓝色渐变到红色
    int r = int(255 * ratio);
    int b = int(255 * (1 - ratio));
    cv::Scalar color(b, 0, r);
    
    // 绘制半透明矩形
    auto p0 = Tf2Img_f(x - cell_size/2, y - cell_size/2);
    auto p1 = Tf2Img_f(x + cell_size/2, y + cell_size/2);
    
    // 创建叠加层实现半透明效果
    cv::Mat overlay = cur_frame_.clone();
    cv::rectangle(overlay, p0, p1, color, -1);
    cv::addWeighted(overlay, 0.3, cur_frame_, 0.7, 0, cur_frame_);
  }
}

void MapDrawer::DrawStatistics(int total_kf, int total_cells, int total_submaps,
                               int rtk_count, int loop_count) {
  if (!is_init_) return;
  
  // 在图像左上角绘制统计信息
  int y_offset = 30;
  int line_height = 25;
  
  char buf[256];
  
  snprintf(buf, sizeof(buf), "KeyFrames: %d", total_kf);
  cv::putText(cur_frame_, buf, cv::Point(10, y_offset), 
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv_black, 1);
  y_offset += line_height;
  
  snprintf(buf, sizeof(buf), "Cells: %d", total_cells);
  cv::putText(cur_frame_, buf, cv::Point(10, y_offset), 
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv_black, 1);
  y_offset += line_height;
  
  snprintf(buf, sizeof(buf), "SubMaps: %d", total_submaps);
  cv::putText(cur_frame_, buf, cv::Point(10, y_offset), 
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv_black, 1);
  y_offset += line_height;
  
  snprintf(buf, sizeof(buf), "RTK: %d", rtk_count);
  cv::putText(cur_frame_, buf, cv::Point(10, y_offset), 
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv_green, 1);
  y_offset += line_height;
  
  snprintf(buf, sizeof(buf), "Loop: %d", loop_count);
  cv::putText(cur_frame_, buf, cv::Point(10, y_offset), 
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv_cyan, 1);
  y_offset += line_height;
  
  // 绘制图例
  y_offset += 10;
  cv::putText(cur_frame_, "Direction:", cv::Point(10, y_offset), 
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv_black, 1);
  y_offset += 20;
  
  const char* dir_labels[] = {"0-60", "60-120", "120-180", "180-240", "240-300", "300-360"};
  for (int i = 0; i < 6; i++) {
    cv::Scalar color = GetDirectionColor(i);
    cv::rectangle(cur_frame_, cv::Point(10, y_offset - 10), 
                  cv::Point(25, y_offset + 5), color, -1);
    cv::putText(cur_frame_, dir_labels[i], cv::Point(30, y_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv_black, 1);
    y_offset += 18;
  }
}