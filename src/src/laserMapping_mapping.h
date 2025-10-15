#ifndef LASERMAPPING_MAPPING_H
#define LASERMAPPING_MAPPING_H 

#include "laserMapping_help.h"
#include <std_srvs/SetBool.h> 

bool loadExistingMap();
void exportStaticMapExample( );
bool save_map_cbk(std_srvs::SetBool::Request &req,std_srvs::SetBool::Response &res);
void save_map_accumulated_cloud();
void save_map_PclWaitSave();

#endif