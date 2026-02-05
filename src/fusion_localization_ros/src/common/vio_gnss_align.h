#ifndef COMMON_VIO_GNSS_ALIGN_H_
#define COMMON_VIO_GNSS_ALIGN_H_

#include <vector>
#include "common/data_utils.h"
#include "common/data_type_basic.h"
#include "common/data_type.h"

struct SpaNode {
  double timestamp = 0.0;
  common::ProbPose pose;                          // origin vio/lio
  common::ProbPose align_pose;                    // aligned vio/lio, 边缘化
  std::shared_ptr<common::ProbPose> reloc_ref;    // vio_reloc/lio_reloc
  std::shared_ptr<common::ProbPose> gnss_ref;     // gnss
};

struct SpaConfig {
  bool use_align_pose = true;
  double pose_adj_factor = 1.0;
  double pose_align_factor = 0.16;
  double pose_rp_factor = 0.04;
};

// 用于初始化tf_pose后的spa
common::Data_Pose spa_align(std::vector<SpaNode> &vv_vec, const SpaConfig &config);

// 用于建图的无RTK下桩
common::Data_Pose vio_gnss_init(const std::vector<SpaNode> &vv_vec);

#endif  // COMMON_VIO_GNSS_ALIGN_H_