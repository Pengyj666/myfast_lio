#ifndef LIO_HELPER_H
#define LIO_HELPER_H

#include <mutex>
#include <thread>
#include <deque>
#include <atomic>
#include <Eigen/Core>
#include <sensor_msgs/Imu.h>
#include <pcl/filters/voxel_grid.h>

#include "ikd_Tree.h"
#include "common_lib.h"
#include "use-ikfom.hpp"
#include "preprocess.h"
#include "IMU_Processing.h"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)

class LioHelper {
public:
    LioHelper();
    ~LioHelper() = default;

    mutex mtx_buffer;
    int num_max_iterations = 0;
    bool extrinsic_est_en = true;
    float DET_RANGE = 50.0f;
    const float MOV_THRESHOLD = 0.2f;
    string root_dir = ROOT_DIR;

    double gyr_cov = 0.1;
    double acc_cov = 0.1;
    double b_gyr_cov = 0.0001;
    double b_acc_cov = 0.0001;
    double cube_len = 0;

    bool point_selected_surf[100000] = {0};
    vector<vector<int>> pointSearchInd_surf;
    vector<PointVector> Nearest_Points;
    vector<double> extrinT;
    vector<double> extrinR;

    V3D Lidar_T_wrt_IMU{Zero3d};
    M3D Lidar_R_wrt_IMU{Eye3d};

    
    double last_timestamp_imu = -1.0;
    double lidar_end_time = 0.0;
    double first_lidar_time = 0.0;
    int feats_down_size = 0;
    double lidar_mean_scantime = 0.0;
    int scan_num = 0;
    int lidar_type = 5;
    bool lidar_pushed = false;
    bool flg_first_scan = true;
    bool flg_EKF_inited = false;
    bool Localmap_Initialized = false;

    vector<BoxPointType> cub_needrm;

    deque<double> time_buffer;
    deque<PointCloudXYZI::Ptr> lidar_buffer;
    deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
    KD_TREE<PointType> ikdtree;

    shared_ptr<ImuProcess> p_imu;

    BoxPointType LocalMap_Points;

    PointCloudXYZI::Ptr featsFromMap;
    PointCloudXYZI::Ptr feats_undistort;
    PointCloudXYZI::Ptr feats_down_body;
    PointCloudXYZI::Ptr feats_down_world;
    PointCloudXYZI::Ptr normvec;
    PointCloudXYZI::Ptr laserCloudOri;
    PointCloudXYZI::Ptr corr_normvect;
    PointCloudXYZI::Ptr _featsArray;

    MeasureGroup Measures;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    state_ikfom state_point;
    vect3 pos_lid;

    double filter_size_map_min = 0;
    double filter_size_surf_min = 0;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
    void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
    void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po,double lidar_d, vector<double> translation_body);
    void pointBodyToWorld(PointType const * const pi, PointType * const po);
    void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s);
    void init();
    void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    bool sync_packages();
    void lasermap_fov_segment();
    void map_incremental();
    bool imu_pretreatment();
    void init_imu_extrin();
    bool init_kdtree();
    void save_PCL_Storage();
    void resizePointCloud();
    void reset();

    void downSizeFilter();
    void regSaveMapPointCallback(std::function<void(PointVector&, PointVector&)> cb,std::function<void(void)> cb_clear_map);
    void regPubOdomCallback(std::function<void(void)> cb_pub_odom_);
    void regPubPointCloudCallback(std::function<void(void)> cb_pub_point_cloud_);
    void regSetGeoQuatCallback(std::function<void(state_ikfom&)> cb_set_geoQuat_);

    static void h_share_model_static(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    static LioHelper* static_instance;
    
    state_ikfom get_state_point(){ return state_point; };
    void set_state_point(state_ikfom s){ state_point = s; };
    void inset_lidar_buffer(PointCloudXYZI::Ptr pcl_in){ lidar_buffer.push_back(pcl_in); };
    void clear(){lidar_buffer.clear();};

    std::function<void(void)> cb_pub_odom;
    std::function<void(void)> cb_pub_point_cloud;
    std::function<void(state_ikfom&)> cb_set_geoQuat;
private:
    std::function<void(PointVector&,PointVector&)> cb_save_map;
    std::function<void(void)> cb_clear_map;

};

#endif // LIO_HELP_H