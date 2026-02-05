#ifndef VMAP_MAP_DRAWER_H
#define VMAP_MAP_DRAWER_H

#include <atomic>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>

// map img
//         ddr.x
//         ^
//         |
// ddr.y<--.----------> u. cv.x
//         |         | 
//         | map-img |
//         |---------.
//         V
//         v. cv.y
// affine mat : r0 r1 t0
//              r2 r3 t1
// map-pose = (x, y, th) >> img-pose = (u, v, th')
// u = r0*x + r1*y + t0
// v = r2*x + r3*y + t1
// th: [+x -> +y -> -x] = [0 ->  pi/2 ->  pi]
//     [+x -> -y -> -x] = [0 -> -pi/2 -> -pi]
class MapDrawer {
 public:
  struct TimedPose {
    double timestamp = 0.0;
    float xyz[3] = {0.f, 0.f, 0.f};
    float rpy[3] = {0.f, 0.f, 0.f};
  };
  struct TrajConfig {
    int cc_bar_type = 0;  // 0: sigle-color, 1: color_bar
    cv::Scalar color = cv::Scalar(0, 255, 0);
  };
  struct CanvasParams {
    int width = 800;
    int height = 800;
    float resolution = 0.05;  // meter per pixel
    float org_xy[2] = {400.f, 400.f};
  };
  
  // 空间索引可视化用的关键帧信息
  struct KeyFrameVis {
    float x = 0.f;
    float y = 0.f;
    float yaw = 0.f;           // 弧度
    int direction_slot = 0;    // 0-5
    int cell_x = 0;
    int cell_y = 0;
    int submap_x = 0;
    int submap_y = 0;
    bool has_rtk = false;      // 是否有RTK约束
    bool has_loop = false;     // 是否有回环
  };

  MapDrawer();
  void InitCanvas(const CanvasParams& canvas_params);

  cv::Mat GetMap() const { return cur_frame_.clone(); }
  
  void DrawTraj(const std::vector<TimedPose>& traj, const TrajConfig& config);

  void DrawGrid();
  // 绘制原点
  void DrawOrgP();
  
  // ========== 空间索引可视化 ==========
  
  // 绘制所有关键帧（带方向箭头，按方向槽位着色）
  void DrawKeyFrames(const std::vector<KeyFrameVis>& keyframes);
  
  // 绘制 Cell 网格（0.25m）
  void DrawCellGrid(float cell_size = 0.25f);
  
  // 绘制 SubMap 边界（5m）
  void DrawSubMapGrid(float submap_size = 5.0f);
  
  // 绘制 Cell 占用热力图（颜色深浅表示关键帧数量）
  void DrawCellHeatmap(const std::vector<KeyFrameVis>& keyframes, float cell_size = 0.25f);
  
  // 绘制统计信息文字
  void DrawStatistics(int total_kf, int total_cells, int total_submaps,
                      int rtk_count, int loop_count);
  
  // 获取方向槽位对应的颜色（6种颜色）
  static cv::Scalar GetDirectionColor(int direction_slot);

 private:
  cv::Point2f Tf2Img_f(float x, float y);
  cv::Point2i Tf2Img_i(float x, float y);

  std::atomic_bool is_init_;
  CanvasParams canvas_params_;

  cv::Mat canvas_, cur_frame_;
  float affine_mat_[2][3];
};

#endif//VMAP_MAP_DRAWER_H