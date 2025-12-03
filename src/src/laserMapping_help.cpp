#include <mutex>
#include <thread>
#include "laserMapping_help.h"


/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = true, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;

/**************************/

float res_last[100000] = {0.0};
// 检测范围定义，用于确定物体检测的最大距离
float DET_RANGE = 50.0f;

// 移动阈值常量，用于判断物体是否发生移动的最小距离标准
const float MOV_THRESHOLD = 0.2f;
double time_diff_lidar_to_imu = 0.0;


condition_variable sig_buffer;

PointCloudXYZI::Ptr accumulated_cloud(new PointCloudXYZI());
mutex accumulated_cloud_mutex;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
int    txt_save_interval = 2; // 保存txt文件的间隔
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

std::atomic<bool> save_map = {false};

/**
 * @brief 从地图中提取的特征点云
 * 
 * 该点云存储从地图中提取的原始特征点，包含XYZ坐标和强度信息
 */
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());

/**
 * @brief 去畸变后的特征点云
 * 
 * 该点云存储经过运动畸变校正后的特征点，用于提高配准精度
 */
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());

/**
 * @brief 降采样后的机体坐标系特征点云
 * 
 * 该点云存储在机体坐标系下经过降采样处理的特征点，减少计算量
 */
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());

/**
 * @brief 降采样后的世界坐标系特征点云
 * 
 * 该点云存储转换到世界坐标系并降采样后的特征点，用于全局匹配
 */
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());

/**
 * @brief 法向量点云
 * 
 * 该点云存储特征点的法向量信息，容量预分配为100000个点
 * 用于点云配准中的几何约束计算
 */
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));

/**
 * @brief 原始激光点云
 * 
 * 该点云存储原始的激光雷达数据点，容量预分配为100000个点
 * 用于后续的特征提取和处理
 */
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));

/**
 * @brief 对应法向量点云
 * 
 * 该点云存储与特征点对应的法向量信息，容量预分配为100000个点
 * 用于优化算法中的约束条件计算
 */
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));

/**
 * @brief 特征点云数组
 * 
 * 该指针用于存储特征点云的数组结构，具体分配在后续代码中进行
 */
PointCloudXYZI::Ptr _featsArray;



// ikd-Tree树用于增量最近邻搜索
KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
// 扩展卡尔曼滤波器的测量数据组，用于存储传感器测量信息
MeasureGroup Measures;
// 扩展卡尔曼滤波器实例，模板参数为状态类型、噪声维度和输入类型
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
// EKF状态点，存储当前估计的状态信息
state_ikfom state_point;
// 激光雷达位置向量
vect3 pos_lid;


// 点云预处理模块的智能指针
shared_ptr<Preprocess> p_pre(new Preprocess());
// IMU数据处理模块的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess());

PointCloudXYZI::Ptr down_map(new PointCloudXYZI()); 

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po )
{
    V3D p_body(pi->x, pi->y, pi->z);
    // p_body是激光雷达坐标系 -IMU坐标系 - 世界坐标系 转换到世界坐标系
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po )
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

}

void pointBodyToWorld(PointType const * const pi, PointType * const po )
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}




void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}



/**
 * @brief 共享模型函数，用于计算激光点与地图中最近表面之间的残差，并构建EKF更新所需的观测矩阵H和观测向量。
 * 
 * 该函数的主要任务包括：
 * 1. 将激光点从体坐标系转换到世界坐标系；
 * 2. 搜索每个点在地图中的最近邻点并估计局部平面；
 * 3. 计算点到平面的距离作为观测残差；
 * 4. 构建观测雅可比矩阵（H）和观测向量（h）供EKF使用。
 *
 * @param s 当前状态估计结构体，包含位姿、偏移等信息。
 * @param ekfom_data EKF共享数据结构，用于传递观测相关数据（如H矩阵、观测值等）。
 */
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** 最近表面搜索与残差计算 **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* 坐标变换：将点从体坐标系转换到世界坐标系 */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** 在地图中查找最近的表面点 **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.07f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    // cout<<"omp_get_wtime() -match_start): "<< static_cast<long long>((omp_get_wtime() -match_start)* 1e6) <<endl;;
    double solve_start_  = omp_get_wtime();
    
    /*** 构建观测雅可比矩阵H和观测向量 ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** 获取最近表面的法向量 ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** 计算观测雅可比矩阵H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** 观测值：点到最近表面的距离 ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}



double lidar_mean_scantime = 0.0;
int    scan_num = 0;
/**
 * @brief 同步激光雷达和IMU数据包，确保时间戳对齐
 * 
 * 该函数用于同步激光雷达扫描数据与IMU数据。首先检查激光雷达和IMU缓冲区是否为空，
 * 若不为空，则将激光雷达数据加入测量组，并根据点云数据计算扫描结束时间。
 * 然后从IMU缓冲区中提取时间早于激光雷达结束时间的数据，一并加入测量组。
 * 
 * @param Measures 测量组引用，用于存储同步后的激光雷达和IMU数据
 * @return bool 返回true表示成功同步一组数据，返回false表示数据不足或未满足时间条件
 */
bool sync_packages()
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }
 
    /*** 处理激光雷达数据：将前端的激光雷达扫描数据加入测量组 ***/
    if(!lidar_pushed)
    {
        Measures.lidar = lidar_buffer.front();
        Measures.lidar_beg_time = time_buffer.front();

        // 根据点云数量和最后一个点的曲率信息估算扫描结束时间
        if (Measures.lidar->points.size() <= 1) // 点数太少，使用平均扫描时间
        {
            lidar_end_time = Measures.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (Measures.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = Measures.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = Measures.lidar_beg_time + Measures.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (Measures.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        // 如果是MARSIM类型激光雷达，扫描结束时间等于开始时间
        if(lidar_type == MARSIM)
            lidar_end_time = Measures.lidar_beg_time;

        Measures.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    // 检查最新的IMU时间戳是否已经覆盖到激光雷达扫描结束时间
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** 提取并填充IMU数据：从IMU缓冲区取出时间在lidar_end_time之前的所有数据 ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    Measures.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        Measures.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    

    // 弹出已处理的激光雷达数据
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

/**
 * @brief 收集点云缓存数据
 * 
 * 该函数用于从ikdtree中获取被移除的点云数据，并将其存储到历史记录中。
 * 可以用于后续的点云数据处理或分析。
 * 
 * @note 该函数不接受任何参数，无返回值
 */
void points_cache_collect()
{
    PointVector points_history;
    // 从ikdtree中获取被移除的点云数据
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
/**
 * @brief 根据激光雷达的视场角（FOV）对局部地图进行分段管理，动态更新局部地图边界。
 * 
 * 此函数用于维护一个随传感器位置变化而动态调整的局部地图区域。当传感器靠近当前局部地图边界时，
 * 会根据设定的阈值移动局部地图区域，并删除旧区域中的点云数据以保持地图更新。
 * 
 * 主要功能包括：
 * - 初始化局部地图区域；
 * - 判断是否需要移动局部地图；
 * - 计算新的局部地图边界；
 * - 删除旧区域中的点云数据。
 * 
 * @note 该函数不接受参数，也不返回任何值。
 */
void lasermap_fov_segment()
{
    // 清空需要删除的立方体列表
    cub_needrm.clear();
    
    // 重置KD树删除计数器和时间统计
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    

    // 将X轴方向的点从机体坐标系转换到世界坐标系
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);

    // 获取当前激光雷达的位置
    V3D pos_LiD = pos_lid;

    // 如果局部地图尚未初始化，则初始化局部地图边界并返回
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 计算当前位置到局部地图各边的距离，并判断是否需要移动局部地图
    float dist_to_map_edge[3][2];
    bool need_move = false;
    float effective_threshold = min(MOV_THRESHOLD * DET_RANGE, (float)(cube_len * 0.3));
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);

        if (dist_to_map_edge[i][0] <= effective_threshold || dist_to_map_edge[i][1] <= effective_threshold) need_move = true;
    }

    // 如果不需要移动局部地图，则直接返回
    if (!need_move) return;

    // 定义新的局部地图边界和临时边界变量
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;

    // 计算移动距离
    float mov_dist = max((cube_len - 2.0 * effective_threshold) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    cout << "mov_dist " << mov_dist << endl;
    // 根据距离判断是否需要移动局部地图的各个边界，并记录需要删除的区域
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= effective_threshold){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= effective_threshold){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }

    // 更新局部地图边界
    LocalMap_Points = New_LocalMap_Points;

    // 收集需要缓存的点云数据
    //points_cache_collect();

    // 删除旧区域中的点云数据，并统计删除时间和删除点数
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}


int process_increments = 0;
/**
 * @brief 增量式构建地图点云，对输入的特征点进行坐标变换、下采样判断，并将需要添加的点插入到KD树中。
 *
 * 该函数的主要流程包括：
 * 1. 将点从体坐标系（body frame）转换到世界坐标系（world frame）；
 * 2. 判断每个点是否需要加入地图（通过近邻点距离和下采样策略）；
 * 3. 将需要添加的点增量式地插入到ikd-tree中。
 *
 * @note 该函数不返回任何值，但会更新全局变量：ikdtree、add_point_size、kdtree_incremental_time。
 */
void map_incremental( )
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    // 遍历所有降采样后的特征点
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        /* decide if need add to map  决定是否需要添加到地图*/
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 

            // 计算当前点所在的下采样网格中心点
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;

            float dist  = calc_dist(feats_down_world->points[i],mid_point);

            // 增加距离阈值判断，避免添加过多近邻点
            if (dist > filter_size_map_min * 0.8) {
                PointToAdd.push_back(feats_down_world->points[i]);
                continue;
            }

            // 如果最近邻点与当前网格中心距离过大，则不需要下采样，直接保留
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }

            // 检查是否有更靠近网格中心的点存在，若有则不添加当前点
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }

            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            // 如果没有近邻点或EKF未初始化，则直接添加该点
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    // 将筛选后的点增量式地添加到KD树中并统计时间
    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, false);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    //add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;

    if(save_map.load()){
        {
            std::lock_guard<std::mutex> lock(accumulated_cloud_mutex);
            for (const auto& point : PointToAdd) {
                accumulated_cloud->push_back(point);
            }
            for (const auto& point : PointNoNeedDownsample) {
                accumulated_cloud->push_back(point);
            }
        }
    }
}

bool  imu_pretreatment(){
    bool result = true;
    if(flg_first_scan){
        p_imu->Reset();
        Measures.imu.clear();
        flg_first_scan = false;
        result = false;   
    }

    // IMU预积分处理
    p_imu->Process(Measures, kf, feats_undistort);

    state_point = kf.get_x();

    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    if (feats_undistort->empty() || (feats_undistort == NULL))
    {
        ROS_WARN("IMU No point, skip this scan!\n");
        first_lidar_time = Measures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        result = false; 
    }

    flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                    false : true;
    kdtree_search_time = 0.0;
    match_time = 0;
    solve_time = 0;
    solve_const_H_time = 0;

    return result;
}


void init_imu_extrin(){    
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
}

bool init_kdtree(){
    bool result = true;
    /*** 初始化地图kdtree ***/
    if(ikdtree.Root_Node == nullptr)
    {
        if(feats_down_size > 5)
        {
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);
            for(int i = 0; i < feats_down_size; i++)
            {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Build(feats_down_world->points);
        }
        result = false;
    }
    kdtree_size_st = ikdtree.size();
    return result;
}

void save_PCL_Storage(){ 
    PointVector ().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
    featsFromMap->clear();
    featsFromMap->points = ikdtree.PCL_Storage;
    //cout<<"[map_update]: map size: "<<featsFromMap->points.size()<<endl;
}

void resizePointCloud(){
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);

    SO3ToEuler(state_point.offset_R_L_I);
    //是否要查看地图
    if(0) save_PCL_Storage();

    pointSearchInd_surf.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);
}


#pragma region 日志文件
FILE *fp;
ofstream fout_pre, fout_out, fout_dbg;
void init_pos_log(){

    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;
}

void close_pos_log(){
    if (fp) fclose(fp);
    if (fout_pre) fout_pre.close();
    if (fout_out) fout_out.close();
    if (fout_dbg) fout_dbg.close();
}

void debug_runtime_pos_log(double solve_H_time){
    static int frame_num = 0;
    static double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    frame_num ++;
    kdtree_size_end = ikdtree.size();
    // aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
    // aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
    aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
    aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
    aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
    aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
    T1[time_log_counter] = Measures.lidar_beg_time;
    /*

    - `s_plot[time_log_counter]`: **总处理时间** - 从t0到t5的完整处理时间，即`T5 - T0`
        t0: 开始时间 - 整个处理循环开始的时间点
        t1: IMU预处理和降采样完成时间 - 包括IMU预积分处理和点云降采样
        t2: ICP匹配前时间 - 特征匹配前的准备工作完成时间
        t3: 状态估计完成时间 - EKF迭代优化完成的时间
        t4: 未使用 - 代码中没有明显使用t4
        t5: 结束时间 - 地图增量更新完成的时间点

    - `s_plot2[time_log_counter]`: **扫描点数量** - 当前帧去畸变后的点云数量，即`feats_undistort->points.size()`
    - `s_plot3[time_log_counter]`: **KD树增量时间** - 向KD树添加新点的时间，即`kdtree_incremental_time`
    - `s_plot4[time_log_counter]`: **KD树搜索时间** - 在KD树中搜索近邻点的时间，即`kdtree_search_time`
    - `s_plot5[time_log_counter]`: **KD树删除计数** - 从KD树中删除的点的数量，即`kdtree_delete_counter`
    - `s_plot6[time_log_counter]`: **KD树删除时间** - 从KD树删除点所用的时间，即`kdtree_delete_time`
    - `s_plot7[time_log_counter]`: **KD树初始大小** - 处理前KD树中的点数，即`kdtree_size_st`
    - `s_plot8[time_log_counter]`: **KD树最终大小** - 处理后KD树中的点数，即`kdtree_size_end`
    - `s_plot9[time_log_counter]`: **平均消耗时间** - 截至当前帧的平均总处理时间，即`aver_time_consu`
    - `s_plot10[time_log_counter]`: **新增点数** - 本次添加到地图中的点的数量，即`add_point_size`

    */
    // s_plot[time_log_counter] = t5 - t0;
    s_plot2[time_log_counter] = feats_undistort->points.size();
    s_plot3[time_log_counter] = kdtree_incremental_time;
    s_plot4[time_log_counter] = kdtree_search_time;
    s_plot5[time_log_counter] = kdtree_delete_counter;
    s_plot6[time_log_counter] = kdtree_delete_time;
    s_plot7[time_log_counter] = kdtree_size_st;
    s_plot8[time_log_counter] = kdtree_size_end;
    // s_plot9[time_log_counter] = aver_time_consu;
    s_plot10[time_log_counter] = add_point_size;
    time_log_counter ++;
    //printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
    V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
    // fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
    // <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
    dump_lio_state_to_log(fp);
}

void save_runtime_pos_log(){
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
    FILE *fp2;
    string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
    fp2 = fopen(log_dir.c_str(),"w+");
    fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
    for (int i = 0;i<time_log_counter; i++){
        //s_plot11   雷达入参预处理耗时
        fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
        t.push_back(T1[i]);
        s_vec.push_back(s_plot9[i]);
        s_vec2.push_back(s_plot3[i] + s_plot6[i]);
        s_vec3.push_back(s_plot4[i]);
        s_vec5.push_back(s_plot[i]);
    }
    fclose(fp2);
}
/**
 * @brief 将LIO状态信息写入日志文件
 * 
 * 该函数将当前LIO系统的状态信息格式化输出到指定的日志文件中，
 * 包括时间戳、旋转角度、位置、角速度、线速度、加速度、陀螺仪偏置、
 * 加速度计偏置和重力向量等信息。
 * 
 * @param fp 指向日志文件的文件指针，用于写入状态信息
 */
void dump_lio_state_to_log(FILE *fp)  
{
    // 将旋转矩阵转换为欧拉角形式表示的旋转角度
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    
    // 写入相对于首次激光雷达时间的时间戳
    //fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    
    // 写入旋转向量（欧拉角）
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
    
    // 写入位置坐标
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));
    
    // 写入角速度（此处默认为0）
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
    
    // 写入线速度
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));
    
    // 写入加速度（此处默认为0）
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);
    
    // 写入陀螺仪偏置
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));
    
    // 写入加速度计偏置
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));
    
    // 写入重力向量
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]);
    
    // 写入换行符并刷新文件缓冲区
    fprintf(fp, "\r\n");
    fflush(fp);
}

#pragma endregion