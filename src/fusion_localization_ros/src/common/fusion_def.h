#ifndef LOCALIZATION_FUSION_DEF_H
#define LOCALIZATION_FUSION_DEF_H

#include <string>

enum ModeType {
  MODE_HOLDPLACE = 0,
  MODE_MAP = 1,
  MODE_LOC = 2,
};
const std::string& ModeTypeToString(ModeType mode_type);

enum FusionType {
  FUSION_HOLDPLACE = 0,
  FUSION_RTK_VISION = 1,
  FUSION_VISION = 2,
  FUSION_LIDAR = 3,
};
const std::string& FusionTypeToString(FusionType fusion_type);

enum MapperState {
  MAP_STATE_HOLDPLACE = 0,
  MAP_STATE_UNCONFIGURED = 1, // unconfigured
  MAP_STATE_IDLE = 2,     // configured
  MAP_STATE_READY = 3,    // can start mapping
  MAP_STATE_INITING = 4,  // pose initializing
  MAP_STATE_MAPPING = 5,  // mapping
  MAP_STATE_LOOPING = 6,  // loop closing
  MAP_STATE_SAVING = 7,   // saving map
  MAP_STATE_FINISHED = 8, // lost
  MAP_STATE_ERROR = 9,
};
const std::string& MapperStateToString(MapperState mapper_state);

enum MapperSubState {
  MAP_SUB_STATE_HOLDPLACE = 0,
};
const std::string& MapperSubStateToString(MapperSubState mapper_sub_state);

enum LocatorState {
  LOC_STATE_HOLDPLACE = 0,
  LOC_STATE_UNCONFIGURED = 1,   // configured
  LOC_STATE_IDLE = 2,     // configured
  LOC_STATE_READY = 3,    // loaded map
  LOC_STATE_INITING = 4,  // pose initializing
  LOC_STATE_TRACKING = 5, // tracking
  LOC_STATE_ERROR = 6,    // LOST
};
const std::string& LocatorStateToString(LocatorState locator_state);

enum LocatorSubState {
  LOC_SUB_STATE_HOLDPLACE = 0,
  LOC_SUB_STATE_RTK_INITING_HEADING = 1,
};
const std::string& LocatorSubStateToString(LocatorSubState locator_sub_state);

enum FusionError {
  ERROR_HOLDPLACE = 0,
  ERROR_MAP_LOST = 10,
  ERROR_LOC_LOST = 20,
};
const std::string& FusionErrorToString(FusionError fusion_error);

#endif //LOCALIZATION_FUSION_DEF_H