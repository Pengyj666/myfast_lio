#include "lreloc/lreloc_function.h"
#include "lreloc_node.h"
#include "lreloc_TF_fusion.h"
int main(int argc, char** argv) {
    ros::init(argc, argv, "fast_lio_localization");

    ros::NodeHandle nh;
    std::unique_ptr<lrelocTFFusion> transform_fusion = std::make_unique<lrelocTFFusion>();    // 启动reloc tf

    std::unique_ptr<lreloc_node> lreloc_ = std::make_unique<lreloc_node>(nh);
    lreloc_->init(nh);
    lreloc_->run();
    ros::spin(); 
    return 0;
}