#include "laserMapping_mapping.h"
#include "laserMapping_controller.h"

/**
 * @brief 即时点，将点云从雷达坐标系转换到世界坐标系，并保存到点云中
 * 
 */
bool loadExistingMap() {
    string map_file_path = string(ROOT_DIR) + "/PCD/accumulated_map.pcd";
    
    // 检查地图文件是否存在
    ifstream file(map_file_path);
    if (!file.good()) {
        cout << "No existing map found at: " << map_file_path << endl;
        return false;
    }
    file.close();
    
    // 加载点云地图
    PointCloudXYZI::Ptr map_cloud(new PointCloudXYZI());
    if (pcl::io::loadPCDFile<PointType>(map_file_path, *map_cloud) != 0) {
        cout << "Failed to load map from: " << map_file_path << endl;
        return false;
    }
    
    // 创建体素网格滤波器并执行下采样
    pcl::VoxelGrid<PointType> sor; 
    sor.setInputCloud(map_cloud); 
    sor.setLeafSize(0.4, 0.4, 0.4);
    sor.filter(*down_map);


    cout << "Successfully loaded map with " << map_cloud->size() << " points" << endl;
    // ikdtree.set_downsample_param(filter_size_map_min);

    // ikdtree.Build(down_map->points);
    
    return true;
}


/**
 * @brief 在主循环中调用的地图保存函数示例
 * 
 */
void exportStaticMapExample( ) {
    std::lock_guard<std::mutex> lock(accumulated_cloud_mutex);
    #if 1
    int size = feats_undistort->points.size();
    // 创建用于存储世界坐标系下点云的新点云对象
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    // 将每个点从机体坐标系转换到世界坐标系
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&feats_undistort->points[i], \
                            &laserCloudWorld->points[i]);
    }

    // 将转换后的点云累加到等待保存的点云中
    *accumulated_cloud += *laserCloudWorld;


    #endif
}

/**
 * @brief 保存点云地图的回调函数，用于在用户按下Ctrl+S键时保存点云地图。
 * 
 * @param msg 保存地图的标志位，true表示保存地图，false表示取消保存地图。
 */
const std::string k_meta_map_fn = "meta_map.txt";
bool save_map_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res){
      ROS_INFO("===================saveMap_cbk===================== ");

     	std::lock_guard<std::mutex> lock(accumulated_cloud_mutex);
    if(!req.arg.empty()){
        const std::string vmap_version = "V1";
        string map_path = string(ROOT_DIR) + req.arg ;
        if (map_path.back() != '/')
            map_path += "/";

        std::string lmap_path = map_path + "lmap/";
         struct stat info;
        if (!IsDirExisting(lmap_path.c_str())) {
            CreateDir(lmap_path.c_str());
        }
        if (accumulated_cloud->size() > 0 ) {
            bool expected = true;
            save_map.compare_exchange_strong(expected,false);
            
            const std::string new_sm_name = GetCurTimeStamp_Sec();

            string save_map_path = lmap_path + new_sm_name+ ".pcd";
            printf("saveMap(): %s",save_map_path.c_str());
                        
            string all_points_dir(save_map_path);
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(all_points_dir, *accumulated_cloud);

            std::map<std::string, std::string> submaps;
            // 先备份已存meta_map.txt
            std::string meta_map_fn = lmap_path + k_meta_map_fn;
           
            if (!IsDirExisting(meta_map_fn.c_str())) {
                std::ifstream fin(meta_map_fn);
                std::string tmp_version, tmp_name;
                while (fin >> tmp_name >> tmp_version) {
                    if (tmp_version.size() == 2 && tmp_name.size() == 15) {
                        submaps[tmp_name] = tmp_version;
                    } else {
                        break;
                    }
                }
                fin.close();
            }
            submaps[new_sm_name] = vmap_version;
            {
                std::ofstream fout(meta_map_fn);
                for (const auto &it : submaps) {
                fout << it.first << " " << it.second << std::endl;
                ROS_INFO("saveMap(): name=%s, version=%s", it.first.c_str(), it.second.c_str());
                }
                fout.close();
                ROS_INFO("saveMap() done");
            }


            res.result = 1;
            res.message = "Map saved to %s ",save_map_path.c_str() ;
            ROS_INFO( "Map saved to %s", save_map_path.c_str());
            accumulated_cloud->clear();
        }else{
            res.result = 0;
            res.message = "No points to save or map not initialized";
        }
    }
    return true;
}

bool ctrl_mapping_cbk(mower_msgs::TriggerRequest &req,mower_msgs::TriggerResponse &res){
    if(req.arg =="reset_lio" ){
        ROS_INFO("ctrl_mapping_cbk(reset_lio) Resetting LIO ++++++");
        is_running_.store(2);   //重置算法
        res.result = 1;
        res.message = "LIO reset.";
        ROS_INFO_STREAM(res.message);

    }else if(req.arg =="start_mapping" ){
        save_map.store(true);
        res.result = 1;
        res.message = "Started mapping";

        // 初始化里程计文件
        if (!odom_file_initialized.load()) {
            odom_file.open(string(string(ROOT_DIR) + "/odometry_data.txt").c_str());
            if (odom_file.is_open()) {
                // 写入表头
                odom_file << "timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz" << std::endl;
                odom_file_initialized.store(true);
            }
        }
    }else if(req.arg =="stop_mapping"){
        save_map.store(false);
        res.result = 1;
        res.message = "Stopped mapping";
    }else if(req.arg == "open_lio"){
        if(is_running_.load() == 0){         //
            is_running_.store(1);
        }

        res.result = 1;
        res.message = "LIO started successfully.";
        ROS_INFO_STREAM(res.message);
    }else if(req.arg == "close_lio"){
        is_running_.store(2);
        res.result = 1;
        res.message = "LIO stopped and resources released.";
        ROS_INFO_STREAM(res.message);
    }
    return true;
}


void save_map_PclWaitSave()
{ 
    /**************** 保存地图 ****************/
    /* 1. 确保有足够的内存  这个是按照帧率数量保存
    /* 2. pcd保存会严重影响实时性能 **/ 
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
}

void updatePCDHeaderPointCount(const string& file_path) {
    // 读取整个文件来计算点数
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    if (pcl::io::loadPCDFile<PointType>(file_path, *cloud) == 0) {
        // 重新保存文件，PCL会自动更新头部信息
        pcl::PCDWriter writer;
        writer.writeBinary(file_path, *cloud);
        cout << "Updated PCD header with " << cloud->size() << " points" << endl;
    }
}


void save_map_accumulated_cloud()
{
    if (accumulated_cloud->size() > 0 && save_map.load() == false) {

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        // 格式化时间戳字符串
        std::stringstream timestamp_ss;
        // timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3) << ms.count();
        string save_map_path = string(string(ROOT_DIR) + ".pcd");

        string all_points_dir(save_map_path);
        pcl::PCDWriter pcd_writer;
        // cout << "Saving accumulated point cloud to " << all_points_dir << endl;
        pcd_writer.writeBinary(all_points_dir, *accumulated_cloud);


        accumulated_cloud->clear();
    }
}

