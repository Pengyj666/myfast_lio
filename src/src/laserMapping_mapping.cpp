#include "laserMapping_mapping.h"

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

    #if 1
    std::lock_guard<std::mutex> lock(txt_save_mutex);
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
bool save_map_cbk(std_srvs::SetBool::Request &req,std_srvs::SetBool::Response &res){
    std::cout << "Debug: File=" << __FILE__ << ", Line=" << __LINE__ << ", Function=" << __FUNCTION__ << std::endl;

    struct stat info;
    if (stat(string(string(ROOT_DIR) + "PCD/").c_str(), &info) != 0) {
        res.success = false;
        res.message = "Folder not found PCD";
        return true;
    }
    cout<<"req.data"<< req.data<<endl;
    if(req.data){
        save_map = true;
        res.success = true;
    }else{
        save_map = false;
        res.success = false;
        res.message = "Cancel save map";
    }
    cout<<"save map"<< save_map<<endl;
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
    if (accumulated_cloud->size() > 0 && save_map == false) {

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        // 格式化时间戳字符串
        std::stringstream timestamp_ss;
        timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3) << ms.count();
        string save_map_path = string(string(ROOT_DIR) + "PCD/map_") + timestamp_ss.str() + ".pcd";

        string all_points_dir(save_map_path);
        pcl::PCDWriter pcd_writer;
        // cout << "Saving accumulated point cloud to " << all_points_dir << endl;
        pcd_writer.writeBinary(all_points_dir, *accumulated_cloud);


        accumulated_cloud->clear();
    }
}

