#include "common/fusion_def.h"
#include <map>

namespace {
const std::map<ModeType, std::string> m_mode_type {
    {ModeType::MODE_MAP, "FUSION_MAP"},
    {ModeType::MODE_LOC, "FUSION_LOC"},
    {ModeType::MODE_HOLDPLACE, "FUSION_HOLDPLACE"}
    };

const std::map<FusionType, std::string> m_fusion_type {
    {FusionType::FUSION_RTK_VISION, "FUSION_RTK_VISION"},
    {FusionType::FUSION_VISION, "FUSION_VISION"},
    {FusionType::FUSION_LIDAR, "FUSION_LIDAR"},
    {FusionType::FUSION_HOLDPLACE, "FUSION_HOLDPLACE"}
    };

const std::map<MapperState, std::string> m_mapper_state {
    {MapperState::MAP_STATE_UNCONFIGURED, "MAP_STATE_UNCONFIGURED"},
    {MapperState::MAP_STATE_IDLE, "MAP_STATE_IDLE"},
    {MapperState::MAP_STATE_READY, "MAP_STATE_READY"},
    {MapperState::MAP_STATE_INITING, "MAP_STATE_INITING"},
    {MapperState::MAP_STATE_MAPPING, "MAP_STATE_MAPPING"},
    {MapperState::MAP_STATE_LOOPING, "MAP_STATE_LOOPING"},
    {MapperState::MAP_STATE_SAVING, "MAP_STATE_SAVING"},
    {MapperState::MAP_STATE_FINISHED, "MAP_STATE_FINISHED"},
    {MapperState::MAP_STATE_ERROR, "MAP_STATE_ERROR"},
    {MapperState::MAP_STATE_HOLDPLACE, "MAP_STATE_HOLDPLACE"}
    };

const std::map<MapperSubState, std::string> m_mapper_sub_state {
    {MapperSubState::MAP_SUB_STATE_HOLDPLACE, "MAP_SUB_STATE_HOLDPLACE"}
    };

const std::map<LocatorState, std::string> m_locator_state {
    {LocatorState::LOC_STATE_UNCONFIGURED, "LOC_STATE_UNCONFIGURED"},
    {LocatorState::LOC_STATE_IDLE, "LOC_STATE_IDLE"},
    {LocatorState::LOC_STATE_READY, "LOC_STATE_READY"},
    {LocatorState::LOC_STATE_INITING, "LOC_STATE_INITING"},
    {LocatorState::LOC_STATE_TRACKING, "LOC_STATE_TRACKING"},
    {LocatorState::LOC_STATE_ERROR, "LOC_STATE_ERROR"},
    {LocatorState::LOC_STATE_HOLDPLACE, "LOC_STATE_HOLDPLACE"}
    };

const std::map<LocatorSubState, std::string> m_locator_sub_state {
    {LocatorSubState::LOC_SUB_STATE_HOLDPLACE, "LOC_SUB_STATE_HOLDPLACE"}
    };

const std::map<FusionError, std::string> m_fusion_error {
    {FusionError::ERROR_MAP_LOST, "ERROR_MAP_LOST"},
    {FusionError::ERROR_LOC_LOST, "ERROR_LOC_LOST"},
    {FusionError::ERROR_HOLDPLACE, "ERROR_HOLDPLACE"}
    };
} // namespace

const std::string& ModeTypeToString(ModeType mode_type) {
  if (m_mode_type.count(mode_type) > 0) {
    return m_mode_type.at(mode_type);
  }
  return m_mode_type.at(ModeType::MODE_HOLDPLACE);
}

const std::string& FusionTypeToString(FusionType fusion_type) {
  if (m_fusion_type.count(fusion_type) > 0) {
    return m_fusion_type.at(fusion_type);
  }
  return m_fusion_type.at(FusionType::FUSION_HOLDPLACE);
}

const std::string& MapperStateToString(MapperState mapper_state) {
  if (m_mapper_state.count(mapper_state) > 0) {
    return m_mapper_state.at(mapper_state);
  }
  return m_mapper_state.at(MapperState::MAP_STATE_HOLDPLACE);
}

const std::string& MapperSubStateToString(MapperSubState mapper_sub_state) {
  if (m_mapper_sub_state.count(mapper_sub_state) > 0) {
    return m_mapper_sub_state.at(mapper_sub_state);
  }
  return m_mapper_sub_state.at(MapperSubState::MAP_SUB_STATE_HOLDPLACE);
}

const std::string& LocatorStateToString(LocatorState locator_state) {
  if (m_locator_state.count(locator_state) > 0) {
    return m_locator_state.at(locator_state);
  }
  return m_locator_state.at(LocatorState::LOC_STATE_HOLDPLACE);
}

const std::string& LocatorSubStateToString(LocatorSubState locator_sub_state) {
  if (m_locator_sub_state.count(locator_sub_state) > 0) {
    return m_locator_sub_state.at(locator_sub_state);
  }
  return m_locator_sub_state.at(LocatorSubState::LOC_SUB_STATE_HOLDPLACE);
}

const std::string& FusionErrorToString(FusionError fusion_error) {
  if (m_fusion_error.count(fusion_error) > 0) {
    return m_fusion_error.at(fusion_error);
  }
  return m_fusion_error.at(FusionError::ERROR_HOLDPLACE);
}