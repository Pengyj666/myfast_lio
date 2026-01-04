#include "lreloc_function.h"

lreloc_function::lreloc_function(
                                double localization_th_,
                                std::string map_file_path_,
                                std::string g_map_root_dir_,
                                double map_voxel_size_,
                                double lidar_d_):
    global_map(new pcl::PointCloud<PointT>),
    map_cloud(new pcl::PointCloud<PointT>),
    cur_scan(new pcl::PointCloud<PointT>),
    T_map_to_odom(Eigen::Matrix4f::Identity()){
    localization_th = localization_th_;         //重定位匹配阈值   
    map_file_path = map_file_path_;             //地图文件路径
    g_map_root_dir = g_map_root_dir_;           //地图根目录
    lidar_d = lidar_d_;                         //激光雷达到机器人底盘的高度

    initial_odom_queue.reset(2048);
}

void lreloc_function::init(double localization_th_,
                    std::string map_file_path_,
                    std::string g_map_root_dir_,
                    double map_voxel_size_,
                    double lidar_d_){
    localization_th = localization_th_;         //重定位匹配阈值   
    map_file_path = map_file_path_;             //地图文件路径
    g_map_root_dir = g_map_root_dir_;           //地图根目录
    lidar_d = lidar_d_;                         //激光雷达到机器人底盘的高度
    MAP_VOXEL_SIZE = map_voxel_size_;
}

lreloc_function::~lreloc_function(){
    // global_map.reset();
    // map_cloud.reset();
    // cur_scan.reset();
}

#pragma region 

void lreloc_function::regPubMapToOdomCallback(const std::function<void(std::shared_ptr<Eigen::Matrix4f>)> pub_map_to_odom_) {
    cb_pub_map_to_odom = pub_map_to_odom_;
}


void lreloc_function::insert_initial_odom_queue(Eigen::Matrix4f& initial_odom,double time){
    initial_odom_queue.emplace_back(initial_odom, time);
}

void lreloc_function::set_cur_scan(pcl::PointCloud<PointT>::Ptr& cur_scan_){
    std::lock_guard<std::mutex> lock(cur_scan_mutex);
    *cur_scan = *cur_scan_;
}

#pragma endregion



bool lreloc_function::loadMap(){
    cout<< "----------loadmap----------"<<endl;    
    if(map_cloud && map_cloud->empty()==false){
            cout <<"ma_cloud.size() = " << map_cloud->size() << endl;
        map_cloud->clear();
        map_cloud.reset(new pcl::PointCloud<PointT>);
    }
    if(global_map && global_map->empty()==false){
            cout <<"global_map.size() = " << global_map->size() << endl;
        global_map->clear();
        global_map.reset(new pcl::PointCloud<PointT>);
    }
    cout<< "Loading map from: " << map_file_path << endl;
    // 加载点云地图
    if (pcl::io::loadPCDFile(map_file_path, *map_cloud) < 0) {
        cout << "Failed to load map file: " << map_file_path << endl;
        return false;
    }
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*map_cloud, *map_cloud, indices);

    kdtree_bulid(1, map_cloud);
    global_map = map_cloud;
    is_load_map.store(2);
    cout<<"Global map size: "<<global_map->points.size()<<"---------------------------------"<<endl;
    return true;
}

/**
 * @brief: 匹配当前帧点云与全局地图
 * 
 * @param pose_estimation: 当前帧点云的位姿估计结果
 */
bool lreloc_function::globalLocalization(const Eigen::Matrix4f& pose_estimation) {
    if (!cur_scan || cur_scan->empty()) {
        printf("Empty current scan");
        return false;
    }
    
    if (!global_map || global_map->empty()) {
        printf("Empty global map");
        return false;
    }

    auto tic = std::chrono::high_resolution_clock::now();
    Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
    double best_fitness = std::numeric_limits<double>::max();

    // auto fine_result = pointToPlaneICP(cur_scan, global_map,pose_estimation, 15, 1.0, 1e-4);
    
    // // 精匹配阶段
    // Eigen::Matrix4f T_fine = fine_result.first;
    // // 使用较小体素尺寸进行精细匹配
    auto result_fine = pointToPlaneICP(cur_scan, global_map, pose_estimation, 30,0.1);

    best_fitness = result_fine.second;
    best_transformation = result_fine.first;
    
    auto toc = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

    if (best_fitness < localization_th) { 
        T_map_to_odom = best_transformation;
        cb_pub_map_to_odom(std::make_shared<Eigen::Matrix4f>(T_map_to_odom));

        printf("!!! Global localization success !!!");
        return true;
    } else {
        // ROS_WARN("Not match!!!!");
        printf("fitness score: %f", best_fitness);

        return false;
    }
}

/**
 * @brief 主循环即始化全局定位
 * 
 */
void lreloc_function::run() {
    {
        std::lock_guard<std::mutex> lock(map_mutex); 
        cout<< "---------is_load_map---------"<<is_load_map.load()<<endl;    
        if(is_load_map.load() == 1){        //1- 开启地图加载
            is_load_map.store(3);           //3- 加载中
            loadMap();
            initialized.store(false);
            return;
        }else if(is_load_map.load() != 2){      // 等待地图加载完成
            return;
        }         
    }
    if (pose_received.load() && scan_received.load() && (global_map && global_map->empty()==false)) {
        calculating.store(true) ;
        pose_received.store(false);
        scan_received.store(false); 
    }
    if (!initialized.load() && calculating.load()) { 
        auto t1 = std::chrono::high_resolution_clock::now();
        Eigen::Matrix4f quat;
        {
            std::lock_guard<std::mutex> lock(initial_pose_mutex);
            int idx = initial_odom_queue.findAfter(cur_scan_time.load());
            cout<< "idx: " << idx << endl;
            cout<< "initial_odom_queue.size(): " << initial_odom_queue.size() << endl;

            if (idx >= 0 ) {
                cout << "initial_odom_queue(idx): " << static_cast<long long>(initial_odom_queue(idx)* 1e6 ) << endl;
                cout << "initial_odom_queue[idx]: " << initial_odom_queue[idx] << endl;
                cout << "cur_scan_time = " << static_cast<long long>(cur_scan_time.load()* 1e6 ) << endl;
                quat = initial_odom_queue[idx];
            }else{
                cout << "initial_odom_queue(0): " << static_cast<long long>(initial_odom_queue(0)* 1e6 ) << endl;

                cout << "cur_scan_time = " << static_cast<long long>(cur_scan_time.load()* 1e6 ) << endl;
                cout << "initial_odom_queue[0]: " << initial_odom_queue[0] << endl;
                quat = initial_odom_queue[0];
            }
        }
        initialized.store(globalLocalization(quat)); 
        {
            std::lock_guard<std::mutex> lock(cur_scan_mutex);
            cur_scan->clear();
        }
        calculating.store(false) ;

        auto t2 = std::chrono::high_resolution_clock::now();
        auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        printf("initialization Time: %ld ms", t3.count());
    }
    else if(initialized.load() && calculating.load()){
        auto t1 = std::chrono::high_resolution_clock::now();
        globalLocalization(T_map_to_odom); 
        {
            std::lock_guard<std::mutex> lock(cur_scan_mutex);
            cur_scan->clear();
        }
        calculating.store(false) ;

        auto t2 = std::chrono::high_resolution_clock::now();
        auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        printf("initialization ------------------- Time: %ld ms", t3.count());
    }
}

