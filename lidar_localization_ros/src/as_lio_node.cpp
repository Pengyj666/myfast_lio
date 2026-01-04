
#include "lio_node.h"


int main(int argc, char** argv)
{
    dros_log_func_ptr = utils::LogClient_Log;
    LogClientConfig cfg;
    cfg.log_root_dir = "john_logs";
    cfg.log_sub_dir = "lio_logs";
    cfg.log_file_interval = 3600 * 1000;      // 1 hours
    cfg.log_keep_time = 7 * 24 * 3600 * 1000; // 7 days
    cfg.log_prefix = "lio_";
    LogClient_Init(cfg);

    auto start_time = GetNow_Steady();
    droslog(LogLevel::INFO, "main() 系统已启动时间: %lld sec", start_time/1000);

    // 初始化ROS节点
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;


    std::unique_ptr<lioNode> node_ptr = std::make_unique<lioNode>(nh);
    node_ptr->init();
    node_ptr->start();
    cout<<"as_lio_node_end"<<endl;
    ros::spin(); 


    return 0;
}
