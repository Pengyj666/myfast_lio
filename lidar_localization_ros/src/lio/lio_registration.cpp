#include "lio_helper.h"
#include <omp.h>

void LioHelper::regSaveMapPointCallback(std::function<void(PointVector&, PointVector&)> cb_save_map_,std::function<void(void)> cb_clear_map_){
    cb_save_map = cb_save_map_;
    cb_clear_map = cb_clear_map_;
}

void LioHelper::regPubOdomCallback(std::function<void(void)> cb_pub_odom_){
    cb_pub_odom = cb_pub_odom_; 
}

void LioHelper::regPubPointCloudCallback(std::function<void(void)> cb_pub_point_cloud_){
    cb_pub_point_cloud = cb_pub_point_cloud_;
}
void LioHelper::regSetGeoQuatCallback(std::function<void(state_ikfom&)> cb_set_geoQuat_){
    cb_set_geoQuat = cb_set_geoQuat_;
}
