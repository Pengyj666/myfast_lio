#include "lio_helper.h"
#include <omp.h>
LioHelper* LioHelper::static_instance = nullptr;

LioHelper::LioHelper() :
    featsFromMap(new PointCloudXYZI()),
    feats_undistort(new PointCloudXYZI()),
    feats_down_body(new PointCloudXYZI()),
    feats_down_world(new PointCloudXYZI()),
    normvec(new PointCloudXYZI(100000, 1)),
    laserCloudOri(new PointCloudXYZI(100000, 1)),
    corr_normvect(new PointCloudXYZI(100000, 1)),
    extrinT{0.0, 0.0, 0.0},
    extrinR{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     _featsArray(new PointCloudXYZI()),
    p_imu(new ImuProcess())
{
    static_instance = this;
}
void LioHelper::h_share_model_static(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    if (LioHelper::static_instance != nullptr) {
        LioHelper::static_instance->h_share_model(s, ekfom_data);
    }
}
void LioHelper::init(){
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
}

void LioHelper::downSizeFilter(){
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);
}

void LioHelper::reset(){
    ROS_INFO(" LioHelper::reset()()");
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    p_imu->Reset();
    p_imu-> set_b_first_frame_ (true);

    state_point.pos = MTK::vect<3, double>::Zero();     
    state_point.vel = MTK::vect<3, double>::Zero();       
    state_point.bg = MTK::vect<3, double>::Zero();        
    state_point.ba = MTK::vect<3, double>::Zero();       
    state_point.offset_T_L_I = Lidar_T_wrt_IMU;

    state_point.rot = MTK::SO3<double>(Eigen::Matrix3d::Identity()); 
    state_point.offset_R_L_I = Lidar_R_wrt_IMU;       

    Eigen::Vector3d grav_dir(0.0, 0.0, -1.0);  
    state_point.grav = MTK::S2<double, 98090, 10000, 1>(grav_dir);
    kf.change_x(state_point);  


    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P;
    init_P.setIdentity(); // 初始化为单位矩阵

    kf.change_P(init_P);

    flg_first_scan = true;
    flg_EKF_inited = false;
    lidar_pushed = false;
    first_lidar_time = 0.0;
    lidar_mean_scantime = 0.0;
    scan_num = 0;

    mtx_buffer.lock();
    lidar_buffer.clear();
    imu_buffer.clear();
    time_buffer.clear();
    Measures.imu.clear();
    Measures.lidar.reset();
    Measures.lidar_beg_time = 0.0;
    mtx_buffer.unlock();

    Localmap_Initialized = false;
    cub_needrm.clear();
    laserCloudOri->clear();
    corr_normvect->clear();
    normvec->clear();

    if (ikdtree.Root_Node != nullptr) {
        ikdtree.Reset_Tree();
    }
    if(!feats_undistort->empty() || (feats_undistort != NULL)){
        feats_undistort->clear();
    }


    init_imu_extrin();
    feats_down_body.reset(new pcl::PointCloud<PointType>());
    
    cb_clear_map();
}

void LioHelper::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LioHelper::RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po,double lidar_d, vector<double> translation_body)
{
    const double angle = lidar_d * M_PI / 180.0; 
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);
    
    M3D calibrateTilt_X;
    M3D calibrateTilt_Z;
    calibrateTilt_X << 1, 0, 0,
                    0, cos(angle), sin(angle),
                    0, -sin(angle), cos(angle);

    calibrateTilt_Z << 0, -1, 0,
                1, 0, 0,
                0, 0, 1;
    M3D total_rot_ = calibrateTilt_Z * calibrateTilt_X;

    V3D rotated_point = total_rot_ * p_body_imu + V3D(translation_body[0], translation_body[1], translation_body[2]);

    // 复制旋转后的坐标
    if((rotated_point.x()<= 2.0 && rotated_point.x()>= -1.0)&&
        (rotated_point.y()<= 1.0 && rotated_point.y()>= -1.0)&&
        (rotated_point.z()<= 1.0)){
        po->x = rotated_point.x();
        po->y = rotated_point.y();
        po->z = rotated_point.z();
        po->intensity = pi->intensity;
    }
}

void LioHelper::RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LioHelper::pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LioHelper::pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LioHelper::h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

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
            }
        }
    }
    
    int effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    
    double solve_start_ = omp_get_wtime();
    
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
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

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        ekfom_data.h(i) = -norm_p.intensity;
    }
}

bool LioHelper::sync_packages()
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }
 
    if(!lidar_pushed)
    {
        Measures.lidar = lidar_buffer.front();
        Measures.lidar_beg_time = time_buffer.front();

        if (Measures.lidar->points.size() <= 1)
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

        if(lidar_type == MARSIM)
            lidar_end_time = Measures.lidar_beg_time;

        Measures.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    Measures.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        Measures.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void LioHelper::lasermap_fov_segment()
{
    cub_needrm.clear();
    V3D pos_LiD = pos_lid;

    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2];
    bool need_move = false;
    float effective_threshold = min(MOV_THRESHOLD * DET_RANGE, (float)(cube_len * 0.3));
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);

        if (dist_to_map_edge[i][0] <= effective_threshold || dist_to_map_edge[i][1] <= effective_threshold) need_move = true;
    }

    if (!need_move) return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;

    float mov_dist = max((cube_len - 2.0 * effective_threshold) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    cout << "mov_dist " << mov_dist << endl;
    
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

    LocalMap_Points = New_LocalMap_Points;

    if(cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);
}

void LioHelper::map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++)
    {
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (feats_down_world->points[i].z > 10 || feats_down_world->points[i].z < -10)  continue;

        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType mid_point; 

            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;

            float dist  = calc_dist(feats_down_world->points[i],mid_point);

            // if (dist > filter_size_map_min * 0.8) {                  //限制距离太远的不加入
            //     PointToAdd.push_back(feats_down_world->points[i]);
            //     continue;
            // }

            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }

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
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, true); 
    if(cb_save_map)    cb_save_map(PointToAdd, PointNoNeedDownsample);
}

bool LioHelper::imu_pretreatment()
{
    bool result = true;
    if(flg_first_scan){
        p_imu->Reset();
        Measures.imu.clear();
        flg_first_scan = false;
        result = false;   
    }

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

    flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
    return result;
}

void LioHelper::init_imu_extrin()
{    
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
    kf.init_dyn_share(get_f, df_dx, df_dw,LioHelper::h_share_model_static, num_max_iterations, epsi);
}

bool LioHelper::init_kdtree()
{
    bool result = true;
    if(ikdtree.Root_Node == nullptr){
        if(feats_down_size > 5)        {
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);
            for(int i = 0; i < feats_down_size; i++){
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Build(feats_down_world->points);
            cout << "ikdtree size: " << ikdtree.size() << endl;
        }
        result = false;
        cout<<"Init feats_down_size -----------------------" << feats_down_size<<endl;
    }
    return result;
}

void LioHelper::save_PCL_Storage()
{ 
    PointVector ().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
    featsFromMap->clear();
    featsFromMap->points = ikdtree.PCL_Storage;
}

void LioHelper::resizePointCloud()
{
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);

    SO3ToEuler(state_point.offset_R_L_I);
    if(0) save_PCL_Storage();

    pointSearchInd_surf.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);
}