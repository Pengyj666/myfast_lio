#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

/**
 * @brief Preprocess类的构造函数
 * @details 初始化预处理类的各种参数，包括激光雷达类型、盲区距离、点云过滤参数等
 *          同时计算一些角度相关的余弦值用于后续的点云处理
 */
Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;           // 无穷远点的距离边界阈值
  N_SCANS   = 6;            // 激光雷达扫描线数
  SCAN_RATE = 10;           // 扫描频率(Hz)
  group_size = 8;           // 点云分组大小，用于点云聚类分析
  disA = 0.01;              // 点间距离阈值A
  disA = 0.1;               // B?
  p2l_ratio = 225;          // 点到线距离与点间距离的比率阈值
  limit_maxmid =6.25;       // 最大中点距离限制
  limit_midmin =6.25;       // 中点最小距离限制
  limit_maxmin = 3.24;      // 最大最小点距离限制
  jump_up_limit = 170.0;    // 向上跳跃点的距离阈值
  jump_down_limit = 8.0;    // 向下跳跃点的距离阈值
  cos160 = 160.0;           // 用于计算角度的余弦值阈值(160度对应的余弦值)
  edgea = 2;                // 边缘点检测参数a
  edgeb = 0.1;              // 边缘点检测参数b
  smallp_intersect = 172.5; // 小平面相交角度阈值
  smallp_ratio = 1.2;       // 小平面比率阈值
  given_offset_time = false; // 是否给定时间偏移量的标志位

  // 将角度限制值转换为余弦值，用于后续的角度比较计算
  jump_up_limit = cos(jump_up_limit/180*M_PI);
  jump_down_limit = cos(jump_down_limit/180*M_PI);
  cos160 = cos(cos160/180*M_PI);
  smallp_intersect = cos(smallp_intersect/180*M_PI);
}

Preprocess::~Preprocess() {}

/**
 * @brief 设置预处理参数
 * 
 * @param feat_en 特征处理使能标志，true表示启用特征处理，false表示禁用
 * @param lid_type 激光雷达类型，用于区分不同型号激光雷达的数据处理方式
 * @param bld 盲区距离阈值，小于该距离的点云数据将被过滤掉
 * @param pfilt_num 点云滤波数量，每隔指定数量的点保留一个点进行稀疏化处理
 */
void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}

/**
 * @brief 处理Livox点云数据消息，生成预处理后的点云输出
 * 
 * 该函数接收来自Livox激光雷达的自定义消息格式，通过avia_handler进行数据处理，
 * 最终将处理后的点云数据赋值给输出参数pcl_out。
 * 
 * @param msg 输入的Livox自定义消息指针，包含原始点云数据
 * @param pcl_out 输出的点云数据指针，存储处理后的点云信息
 */
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{  
  avia_handler(msg);  // 调用avia处理器对输入消息进行解析和处理
  *pcl_out = pl_surf; // 将处理后的点云数据赋值给输出参数
}

/**
 * @brief 处理点云数据，根据时间单位和激光雷达类型进行相应的数据预处理
 * @param msg 输入的点云数据消息指针
 * @param pcl_out 输出处理后的点云数据指针
 */
void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{

  // 根据时间单位设置时间缩放因子
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  // 根据激光雷达类型调用相应的处理函数
  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;

  case MARSIM:
    sim_handler(msg);
    break;
  case WLR722F:
  //cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << endl;
    WLR722F_handler(msg);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}


/**
 * @brief 处理 vanjee722f 激光雷达点云数据，提取特征点并为后续卡尔曼滤波做准备
 * 
 * 该函数将 sensor_msgs::PointCloud2 消息转换为自定义点类型，按距离和点过滤参数筛选，
 * 并根据扫描线分组，提取平面点和角点等特征，最终填充 pl_surf 和 pl_corn。
 * 
 * @param msg 输入的 vanjee722f 点云消息指针
 */
void Preprocess::WLR722F_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    #if 1
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<WLR722F_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;

    if (pl_orig.points[plsize - 1].timestamp <= 0) return;

    if(feature_enabled)
    {
        // 清空并预留空间到 pl_buff 中，为每个扫描线准备点云缓存
        for (int i = 0; i < N_SCANS; i++)
        {
            pl_buff[i].clear();
            pl_buff[i].reserve(plsize);
            typess[i].clear(); // 确保typess也被清空
        }

        // 遍历所有原始点云数据，按扫描线分发到 pl_buff，并设置点的时间戳（curvature）
        for (int i = 0; i < plsize; i++)
        {
            PointType added_pt;
            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;
            
            int layer  = pl_orig.points[i].ring - 1; // ring start from 1
            
            if (layer >= N_SCANS || layer < 0) continue;
            
            if (i % point_filter_num != 0) continue;
            
            // // 添加距离过滤
            // if(pl_orig.points[i].x*pl_orig.points[i].x + 
            //    pl_orig.points[i].y*pl_orig.points[i].y + 
            //    pl_orig.points[i].z*pl_orig.points[i].z < (blind * blind)) continue;

            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.curvature = pl_orig.points[i].timestamp * time_unit_scale; 
            pl_buff[layer].points.push_back(added_pt);
        }

        // 对每条扫描线上的点云进行处理，计算相邻点距离等信息，并调用特征提取函数
        for (int j = 0; j < N_SCANS; j++)
        {
            PointCloudXYZI &pl = pl_buff[j];
            int linesize = pl.size();
            
            if (linesize < 2) continue;
            
            vector<orgtype> &types = typess[j];
            types.clear();
            types.resize(linesize); // 确保大小匹配
            linesize--;

            // 计算每个点的距离和与下一个点的距离平方
            for (uint i = 0; i < linesize; i++)
            {
                types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y + pl[i].z * pl[i].z);
                vx = pl[i].x - pl[i + 1].x;
                vy = pl[i].y - pl[i + 1].y;
                vz = pl[i].z - pl[i + 1].z;
                types[i].dista = vx * vx + vy * vy + vz * vz;
            }

            // 最后一个点只计算距离
            if(linesize >= 0) {
                types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + 
                                           pl[linesize].y * pl[linesize].y + 
                                           pl[linesize].z * pl[linesize].z);
            }

            // 调用特征提取函数
            give_feature(pl, types);
        }
    }
    else
    {
        for (int i = 0; i < plsize; i++)
        {
            PointType added_pt;
            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;
            
            // 添加距离过滤
            if(added_pt.x*added_pt.x + added_pt.y*added_pt.y + added_pt.z*added_pt.z <= (blind * blind))
                continue;
                
            if (i % point_filter_num == 0)
            {
                added_pt.x = pl_orig.points[i].x;
                added_pt.y = pl_orig.points[i].y;
                added_pt.z = pl_orig.points[i].z;
                added_pt.intensity = pl_orig.points[i].intensity;
                added_pt.curvature = pl_orig.points[i].timestamp * time_unit_scale;
                pl_surf.points.push_back(added_pt);
            }
        }
    }
#endif

 #if 0
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<WLR722F_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    if (pl_orig.points[plsize - 1].timestamp <= 0) return;

   if(feature_enabled)
{
    // 清空并预留空间到 pl_buff 中，为每个扫描线准备点云缓存
    for (int i = 0; i < N_SCANS; i++)
    {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
    }

    // 遍历所有原始点云数据，按扫描线分发到 pl_buff，并设置点的时间戳（curvature）
    for (int i = 0; i < plsize; i++)
    {
      PointType added_pt;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      
      int layer  = pl_orig.points[i].ring -1; // ring start from 1
      if (layer >= N_SCANS || layer) continue;
      //if(pl_orig.points[i].x*pl_orig.points[i].x+pl_orig.points[i].y*pl_orig.points[i].y+pl_orig.points[i].z*pl_orig.points[i].z < (blind * blind)) continue;

      if (i % point_filter_num != 0) continue;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = pl_orig.points[i].timestamp * time_unit_scale; 
      pl_buff[layer].points.push_back(added_pt);
    }

      // 对每条扫描线上的点云进行处理，计算相邻点距离等信息，并调用特征提取函数
      for (int j = 0; j < N_SCANS; j++)
      {
          PointCloudXYZI &pl = pl_buff[j];
          int linesize = pl.size();
          
          if (linesize < 2) continue;
          vector<orgtype> &types = typess[j];
          types.clear();
          types.resize(linesize);
          linesize--;

          // 计算每个点的距离和与下一个点的距离平方
          for (uint i = 0; i < linesize; i++)
          {
              types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y + pl[i].z * pl[i].z);
              vx = pl[i].x - pl[i + 1].x;
              vy = pl[i].y - pl[i + 1].y;
              vz = pl[i].z - pl[i + 1].z;
              types[i].dista = vx * vx + vy * vy + vz * vz;
          }

          // 最后一个点只计算距离
          types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y + pl[linesize].z * pl[linesize].z);

          // 调用特征提取函数
          give_feature(pl, types);
      }
    }
    // else
    // {
    //   for (int i = 0; i < plsize; i++)
    //   {
    //     PointType added_pt;
    //     // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
    //     added_pt.normal_x = 0;
    //     added_pt.normal_y = 0;
    //     added_pt.normal_z = 0;
    //     added_pt.x = pl_orig.points[i].x;
    //     added_pt.y = pl_orig.points[i].y;
    //     added_pt.z = pl_orig.points[i].z;
    //     added_pt.intensity = pl_orig.points[i].intensity;
    //     added_pt.curvature = pl_orig.points[i].timestamp * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;
        
         

    //     if (i % point_filter_num == 0)
    //     {
    //       if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
    //       {
    //         pl_surf.points.push_back(added_pt);
    //       }
    //     }
    //   }
    // }
 #endif

  //段错误
  #if 0
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    // 使用自定义点类型，字段名需与实际消息一致（如 time 或 timestamp）
    pcl::PointCloud<WLR722F_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    if (plsize == 0) return;

    // 判断是否有时间戳
    given_offset_time = (pl_orig.points[plsize - 1].timestamp > 0);

    // 扫描线相关变量
    double omega_l = 0.361 * SCAN_RATE;
    std::vector<bool> is_first(N_SCANS, true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);
    std::vector<float> yaw_last(N_SCANS, 0.0);
    std::vector<float> time_last(N_SCANS, 0.0);

    if (feature_enabled)
    {
        // 清空并预留空间到 pl_buff
        for (int i = 0; i < N_SCANS; i++)
        {
            pl_buff[i].clear();
            pl_buff[i].reserve(plsize);
        }

        // 按扫描线分发点云，并设置时间戳
        for (int i = 0; i < plsize; i++)
        {
            int layer = pl_orig.points[i].ring;
            if (layer >= N_SCANS) continue;

            double range = pl_orig.points[i].x * pl_orig.points[i].x +
                           pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;
            if (range < blind * blind) continue;

            PointType added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;

            // 时间戳处理
            if (given_offset_time)
            {
                added_pt.curvature = static_cast<float>(pl_orig.points[i].timestamp * time_unit_scale);
            }
            else
            {
                double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
                if (is_first[layer])
                {
                    yaw_fp[layer] = yaw_angle;
                    is_first[layer] = false;
                    added_pt.curvature = 0.0;
                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = added_pt.curvature;
                }
                else
                {
                    if (yaw_angle <= yaw_fp[layer])
                        added_pt.curvature =static_cast<float>( (yaw_fp[layer] - yaw_angle) / omega_l);
                    else
                        added_pt.curvature = static_cast<float>((yaw_fp[layer] - yaw_angle + 360.0) / omega_l);

                    if (added_pt.curvature < time_last[layer])
                        added_pt.curvature += static_cast<float>(360.0 / omega_l);

                    yaw_last[layer] = yaw_angle;
                    time_last[layer] = added_pt.curvature;
                }
            }

            pl_buff[layer].points.push_back(added_pt);
        }

        // 特征提取
        for (int j = 0; j < N_SCANS; j++)
        {
            PointCloudXYZI &pl = pl_buff[j];
            int linesize = pl.size();
            if (linesize < 2) continue;
            vector<orgtype> &types = typess[j];
            types.clear();
            types.resize(linesize);
            linesize--;

            for (uint i = 0; i < linesize; i++)
            {
                types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
                vx = pl[i].x - pl[i + 1].x;
                vy = pl[i].y - pl[i + 1].y;
                vz = pl[i].z - pl[i + 1].z;
                types[i].dista = vx * vx + vy * vy + vz * vz;
            }
            types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);

            give_feature(pl, types); // surf/corn特征点填充
        }
    }
    else
    {
        // 只做距离和点过滤
        for (int i = 0; i < plsize; i++)
        {
            double range = pl_orig.points[i].x * pl_orig.points[i].x +
                           pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;
            if (range < blind * blind) continue;
            if (i % point_filter_num != 0) continue;

            PointType added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.normal_x = 0;
            added_pt.normal_y = 0;
            added_pt.normal_z = 0;
            added_pt.curvature = given_offset_time ? pl_orig.points[i].timestamp * time_unit_scale : 0.0;

            pl_surf.points.push_back(added_pt);
        }
    }
  #endif
  
  //效率低
  #if 0
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<pcl::PointXYZI> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_surf.reserve(plsize);
  for (int i = 0; i < plsize; i++)
  {
    // 距离盲区过滤
    double range = pl_orig.points[i].x * pl_orig.points[i].x +
                   pl_orig.points[i].y * pl_orig.points[i].y +
                   pl_orig.points[i].z * pl_orig.points[i].z;
    if (range < blind * blind) continue;

    // 点过滤
    if (i % point_filter_num != 0) continue;

    PointType added_pt;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;

    pl_surf.points.push_back(added_pt);
  }
  #endif
}

// 卡尔曼滤波处理在 IMU_Processing.hpp 的 ImuProcess::Process 方法中完成，输入为提取后的特征点云和IMU数据。
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;
  // cout<<"plsie: "<<plsize<<endl;

  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for(int i=0; i<N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }
  uint valid_num = 0;
  
  if (feature_enabled)
  {
    for(uint i=1; i<plsize; i++)
    {
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //use curvature as time of each laser points

        bool is_new = false;
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count ++;
    double t0 = omp_get_wtime();
    for(int j=0; j<N_SCANS; j++)
    {
      if(pl_buff[j].size() <= 5) continue;
      pcl::PointCloud<PointType> &pl = pl_buff[j];
      plsize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for(uint i=0; i<plsize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);
      }
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);
      give_feature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else
  {
    for(uint i=1; i<plsize; i++)
    {
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++;
        if (valid_num % point_filter_num == 0)
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points, curvature unit: ms

          if(((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
              || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

/**
 * @brief 处理 Ouster OS1-64 激光雷达点云数据的回调函数
 * 
 * 该函数接收来自 Ouster OS1-64 激光雷达的 PointCloud2 消息，进行预处理操作，
 * 包括点云清洗、特征提取（如启用）等。根据是否启用特征提取功能，将点云分为
 * 表面点（surf points）和角点（corner points），并存储在对应的点云容器中。
 * 
 * @param msg 输入的 sensor_msgs::PointCloud2 类型的点云消息指针
 */
void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  // 清空之前处理的点云数据
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();

  // 将 ROS 消息转换为 PCL 点云格式
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);

  int plsize = pl_orig.size();

  // 预分配内存以提高性能
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);

  if (feature_enabled)
  {
    // 如果启用了特征提取功能

    // 清空每条扫描线的缓冲区，并预分配内存
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    // 对每个点进行处理，按扫描线分类
    for (uint i = 0; i < plsize; i++)
    {
      // 计算点到原点的距离平方
      double range = pl_orig.points[i].x * pl_orig.points[i].x +
                     pl_orig.points[i].y * pl_orig.points[i].y +
                     pl_orig.points[i].z * pl_orig.points[i].z;

      // 距离小于盲区阈值的点被忽略
      if (range < (blind * blind)) continue;

      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;

      // 计算点的 yaw 角度（角度制）
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      // 设置点的时间戳（单位转换为毫秒）
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;

      // 根据点所属的扫描线将其加入对应的缓冲区
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    // 对每条扫描线上的点进行特征分析
    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;

      // 计算相邻点之间的距离平方和范围
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }

      // 最后一个点只计算范围
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);

      // 执行特征提取逻辑
      give_feature(pl, types);
    }
  }
  else
  {
    // 如果未启用特征提取，则仅保留表面点

    // double time_stamp = msg->header.stamp.toSec(); // 可选的时间戳记录

    // 遍历原始点云，按一定间隔过滤点
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      // 忽略距离小于盲区的点
      double range = pl_orig.points[i].x * pl_orig.points[i].x +
                     pl_orig.points[i].y * pl_orig.points[i].y +
                     pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;

      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;

      // 设置点的时间戳（单位：毫秒）
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;

      // 添加到表面点云中
      pl_surf.points.push_back(added_pt);
    }
  }

  // 注释掉的发布函数调用（可能用于调试或后续处理）
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** 这些变量仅在没有给出点时间戳时才起作用***/
    double omega_l = 0.361 * SCAN_RATE;       // 扫描角速度
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

   if(feature_enabled)
{
    // 清空并预留空间到 pl_buff 中，为每个扫描线准备点云缓存
    for (int i = 0; i < N_SCANS; i++)
    {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
    }

    // 遍历所有原始点云数据，按扫描线分发到 pl_buff，并设置点的时间戳（curvature）
    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // 单位：毫秒

        // 如果没有给定时间偏移，则根据点的角度估算时间戳
        if (!given_offset_time)
        {
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957; // 弧度转角度
            if (is_first[layer])
            {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // 曲率计算，考虑角度回绕
            if (yaw_angle <= yaw_fp[layer])
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            }
            else
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            // 处理时间戳回绕问题
            if (added_pt.curvature < time_last[layer])  
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
    }

    // 对每条扫描线上的点云进行处理，计算相邻点距离等信息，并调用特征提取函数
    for (int j = 0; j < N_SCANS; j++)
    {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;

        // 计算每个点的距离和与下一个点的距离平方
        for (uint i = 0; i < linesize; i++)
        {
            types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
            vx = pl[i].x - pl[i + 1].x;
            vy = pl[i].y - pl[i + 1].y;
            vz = pl[i].z - pl[i + 1].z;
            types[i].dista = vx * vx + vy * vy + vz * vz;
        }

        // 最后一个点只计算距离
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);

        // 调用特征提取函数
        give_feature(pl, types);
    }
}
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

void Preprocess::sim_handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pl_surf.clear();
    pl_full.clear();
    pcl::PointCloud<pcl::PointXYZI> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    pl_surf.reserve(plsize);
    for (int i = 0; i < pl_orig.points.size(); i++) {
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < blind * blind) continue;
        Eigen::Vector3d pt_vec;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = 0.0;
        pl_surf.points.push_back(added_pt);
    }
}


/**
 * @brief 对点云数据进行特征提取，识别出平面点、边缘点等不同类型的特征点，并将结果存储到相应的容器中。
 * 
 * 该函数主要完成以下任务：
 * 1. 根据距离盲区过滤掉近处的无效点；
 * 2. 判断局部区域是否构成平面（使用`plane_judge`）并标记为Real_Plane或Poss_Plane；
 * 3. 检测跳跃边缘（Edge_Jump）和边缘平面（Edge_Plane）；
 * 4. 处理小平面（small planes）情况；
 * 5. 最后根据ftype类型将点分类为surf（表面点）或corn（角点）。
 *
 * @param pl 输入的点云数据，类型为pcl::PointCloud<PointType>引用
 * @param types 点的属性信息数组，包含每个点的距离、角度、ftype等信息，vector<orgtype>引用
 */
void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types)
{

  //原方法  运行后导致段错误
  #if 0
  cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << endl;
  int plsize = pl.size();
  int plsize2;
  if(plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  // 跳过距离小于盲区的点
  while(types[head].range < blind)
  {
    head++;
  }

  // Surf特征检测：判断局部区域是否为平面
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  uint i_nex = 0, i2;
  uint last_i = 0; uint last_i_nex = 0;
  int last_state = 0;
  int plane_type;

  for(uint i=head; i<plsize2; i++)
  {
    if(types[i].range < blind)
    {
      continue;
    }

    i2 = i;

    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);
    
    if(plane_type == 1)
    {
      // 将当前组标记为平面点
      for(uint j=i; j<=i_nex; j++)
      { 
        if(j!=i && j!=i_nex)
        {
          types[j].ftype = Real_Plane;
        }
        else
        {
          types[j].ftype = Poss_Plane;
        }
      }
      
      // 如果上一个状态也是平面，则进一步判断是否为边缘平面
      if(last_state==1 && last_direct.norm()>0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if(mod>-0.707 && mod<0.707)
        {
          types[i].ftype = Edge_Plane;
        }
        else
        {
          types[i].ftype = Real_Plane;
        }
      }
      
      i = i_nex - 1;
      last_state = 1;
    }
    else // plane_type != 1
    {
      i = i_nex;
      last_state = 0;
    }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  // Edge特征检测：检测跳跃边缘点
  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for(uint i=head+3; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i].ftype>=Real_Plane)
    {
      continue;
    }

    if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
    Eigen::Vector3d vecs[2];

    for(int j=0; j<2; j++)
    {
      int m = -1;
      if(j == 1)
      {
        m = 1;
      }

      if(types[i+m].range < blind)
      {
        if(types[i].range > inf_bound)
        {
          types[i].edj[j] = Nr_inf;
        }
        else
        {
          types[i].edj[j] = Nr_blind;
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
      vecs[j] = vecs[j] - vec_a;
      
      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if(types[i].angle[j] < jump_up_limit)
      {
        types[i].edj[j] = Nr_180;
      }
      else if(types[i].angle[j] > jump_down_limit)
      {
        types[i].edj[j] = Nr_zero;
      }
    }

    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
     
    }
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
    {
      if(types[i].ftype == Nor)
      {
        types[i].ftype = Wire;
      }
    }
  }

  // 小平面处理：检测并标记小范围内的平面点
  plsize2 = plsize-1;
  double ratio;
  for(uint i=head+1; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }
    
    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }

    if(types[i].ftype == Nor)
    {
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }

      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        if(types[i-1].ftype == Nor)
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  // 最终分类：将点分为surf（表面点）和corn（角点）
  int last_surface = -1;
  for(uint j=head; j<plsize; j++)
  {
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;
      }
    
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      if(last_surface != -1)
      {
        PointType ap;
        for(uint k=last_surface; k<j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j-last_surface);
        ap.y /= (j-last_surface);
        ap.z /= (j-last_surface);
        ap.intensity /= (j-last_surface);
        ap.curvature /= (j-last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }

#endif



# if 1
    int plsize = pl.size();
    int plsize2;
    
    // 添加边界检查
    if(plsize == 0 || types.size() != plsize)
    {
        printf("Error: point cloud size mismatch or empty\n");
        return;
    }
    
    uint head = 0;

    // 跳过距离小于盲区的点
    while(head < plsize && types[head].range < blind)
    {
        head++;
    }
    
    // 检查是否所有点都在盲区
    if(head >= plsize)
    {
        return;
    }

    // Surf特征检测：判断局部区域是否为平面
    plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

    Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
    Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

    uint i_nex = 0, i2;
    uint last_i = 0; uint last_i_nex = 0;
    int last_state = 0;
    int plane_type;

    for(uint i=head; i<plsize2; i++)
    {
        // 添加边界检查
        if(i >= types.size() || types[i].range < blind)
        {
            continue;
        }

        i2 = i;

        plane_type = plane_judge(pl, types, i, i_nex, curr_direct);
        
        if(plane_type == 1)
        {
            // 检查边界
            if(i_nex >= types.size()) break;
            
            // 将当前组标记为平面点
            for(uint j=i; j<=i_nex && j<types.size(); j++)
            { 
                if(j!=i && j!=i_nex)
                {
                    types[j].ftype = Real_Plane;
                }
                else
                {
                    types[j].ftype = Poss_Plane;
                }
            }
            
            // 如果上一个状态也是平面，则进一步判断是否为边缘平面
            if(last_state==1 && last_direct.norm()>0.1)
            {
                double mod = last_direct.transpose() * curr_direct;
                if(mod>-0.707 && mod<0.707)
                {
                    if(i < types.size()) {
                        types[i].ftype = Edge_Plane;
                    }
                }
                else
                {
                    if(i < types.size()) {
                        types[i].ftype = Real_Plane;
                    }
                }
            }
            
            i = i_nex - 1;
            last_state = 1;
        }
        else // plane_type != 1
        {
            i = i_nex;
            last_state = 0;
        }

        last_i = i2;
        last_i_nex = i_nex;
        last_direct = curr_direct;
    }

    // Edge特征检测：检测跳跃边缘点
    plsize2 = plsize > 3 ? plsize - 3 : 0;
    for(uint i=head+3; i<plsize2; i++)
    {
        // 添加边界检查
        if(i >= types.size()) break;
        
        if(types[i].range<blind || types[i].ftype>=Real_Plane)
        {
            continue;
        }

        if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
        {
            continue;
        }

        // 确保索引有效
        if(i >= pl.size() || i-1 >= pl.size() || i+1 >= pl.size()) continue;

        Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
        Eigen::Vector3d vecs[2];

        for(int j=0; j<2; j++)
        {
            int m = -1;
            if(j == 1)
            {
                m = 1;
            }

            // 检查索引有效性
            if(i+m >= types.size() || i+m < 0) continue;
            
            if(types[i+m].range < blind)
            {
                if(types[i].range > inf_bound)
                {
                    types[i].edj[j] = Nr_inf;
                }
                else
                {
                    types[i].edj[j] = Nr_blind;
                }
                continue;
            }

            vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
            vecs[j] = vecs[j] - vec_a;
            
            double norm_a = vec_a.norm();
            double norm_vecs = vecs[j].norm();
            
            if(norm_a > 1e-10 && norm_vecs > 1e-10) {
                types[i].angle[j] = vec_a.dot(vecs[j]) / norm_a / norm_vecs;
                if(types[i].angle[j] < jump_up_limit)
                {
                    types[i].edj[j] = Nr_180;
                }
                else if(types[i].angle[j] > jump_down_limit)
                {
                    types[i].edj[j] = Nr_zero;
                }
            }
        }

        double norm_prev = vecs[Prev].norm();
        double norm_next = vecs[Next].norm();
        if(norm_prev > 1e-10 && norm_next > 1e-10) {
            types[i].intersect = vecs[Prev].dot(vecs[Next]) / norm_prev / norm_next;
            
            if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i-1].dista>1e-16 && types[i].dista>4*types[i-1].dista)
            {
                if(types[i].intersect > cos160)
                {
                    if(edge_jump_judge(pl, types, i, Prev))
                    {
                        types[i].ftype = Edge_Jump;
                    }
                }
            }
            else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>1e-16 && types[i-1].dista>4*types[i].dista)
            {
                if(types[i].intersect > cos160)
                {
                    if(edge_jump_judge(pl, types, i, Next))
                    {
                        types[i].ftype = Edge_Jump;
                    }
                }
            }
            else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
            {
                if(edge_jump_judge(pl, types, i, Prev))
                {
                    types[i].ftype = Edge_Jump;
                }
            }
            else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
            {
                if(edge_jump_judge(pl, types, i, Next))
                {
                    types[i].ftype = Edge_Jump;
                }
            }
            else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
            {
                if(types[i].ftype == Nor)
                {
                    types[i].ftype = Wire;
                }
            }
        }
    }

    // 小平面处理：检测并标记小范围内的平面点
    plsize2 = plsize-1;
    double ratio;
    for(uint i=head+1; i<plsize2; i++)
    {
        // 添加边界检查
        if(i >= types.size() || i-1 >= types.size() || i+1 >= types.size()) continue;
        
        if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
        {
            continue;
        }
        
        if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
        {
            continue;
        }

        if(types[i].ftype == Nor)
        {
            if(types[i-1].dista > types[i].dista)
            {
                ratio = types[i-1].dista / types[i].dista;
            }
            else
            {
                ratio = types[i].dista / types[i-1].dista;
            }

            if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
            {
                if(types[i-1].ftype == Nor)
                {
                    types[i-1].ftype = Real_Plane;
                }
                if(types[i+1].ftype == Nor)
                {
                    types[i+1].ftype = Real_Plane;
                }
                types[i].ftype = Real_Plane;
            }
        }
    }

    // 最终分类：将点分为surf（表面点）和corn（角点）
    int last_surface = -1;
    for(uint j=head; j<plsize; j++)
    {
        // 添加边界检查
        if(j >= types.size()) break;
        
        if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
        {
            if(last_surface == -1)
            {
                last_surface = j;
            }
        
            if(j == uint(last_surface+point_filter_num-1))
            {
                if(j < pl.size()) {
                    PointType ap;
                    ap.x = pl[j].x;
                    ap.y = pl[j].y;
                    ap.z = pl[j].z;
                    ap.intensity = pl[j].intensity;
                    ap.curvature = pl[j].curvature;
                    pl_surf.push_back(ap);
                }

                last_surface = -1;
            }
        }
        else
        {
            if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
            {
                if(j < pl.size()) {
                    pl_corn.push_back(pl[j]);
                }
            }
            if(last_surface != -1)
            {
                PointType ap;
                // 确保索引有效
                if(last_surface < pl.size() && j <= pl.size()) {
                    for(uint k=last_surface; k<j && k<pl.size(); k++)
                    {
                        ap.x += pl[k].x;
                        ap.y += pl[k].y;
                        ap.z += pl[k].z;
                        ap.intensity += pl[k].intensity;
                        ap.curvature += pl[k].curvature;
                    }
                    ap.x /= (j-last_surface);
                    ap.y /= (j-last_surface);
                    ap.z /= (j-last_surface);
                    ap.intensity /= (j-last_surface);
                    ap.curvature /= (j-last_surface);
                    pl_surf.push_back(ap);
                }
            }
            last_surface = -1;
        }
    }
#endif
}


void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  double group_dis = disA*types[i_cur].range + disB;
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr;
  disarr.reserve(20);

  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    disarr.push_back(types[i_nex].dista);
  }
  
  for(;;)
  {
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;
    if(two_dis >= group_dis)
    {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    if((j >= pl.size()) || (i_cur >= pl.size())) break;
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];

    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid)
    {
      leng_wid = lw;
    }
  }


  if((two_dis*two_dis/leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();
    return 0;
  }

  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  if(lidar_type==AVIA)
  {
    double dismax_mid = disarr[0]/disarr[disarrsize/2];
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2];

    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  
  curr_direct << vx, vy, vz;
  curr_direct.normalize();
  return 1;
}

bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  if(nor_dir == 0)
  {
    if(types[i-1].range<blind || types[i-2].range<blind)
    {
      return false;
    }
  }
  else if(nor_dir == 1)
  {
    if(types[i+1].range<blind || types[i+2].range<blind)
    {
      return false;
    }
  }
  double d1 = types[i+nor_dir-1].dista;
  double d2 = types[i+3*nor_dir-2].dista;
  double d;

  if(d1<d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

 
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    return false;
  }
  
  return true;
}
