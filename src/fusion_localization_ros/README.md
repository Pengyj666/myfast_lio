# 融合定位

## 接口协议
### RTK-Vision
### Vision
#### topic
- /as_vio/vio_pose_result: nav_msgs::Odometry: 原始vio位姿
- /as_vmap/reloc_result: nav_msgs::Odometry: 重定位结果
- /as_vmap/reloc_pose: nav_msgs::Odometry: 基于重定位对齐到地图的vio
- /as_vmap/vmap_state: std_msgs::String: 重定位节点状态
#### service
- /as_vio/ctrl: mower_msgs::Trigger: as_vio节点控制
- /as_vmap/savemap: mower_msgs::Trigger: 保存地图
- /as_vmap/loadmap: mower_msgs::Trigger: 加载地图
- /as_vmap/ctrl: mower_msgs::Trigger: as_vmap节点控制

### Lidar
#### topic
- /as_lio/lio: nav_msgs::Odometry: 原始lio位姿
- /as_lio/lio_reloc_pose: nav_msgs::Odometry: 基于重定位对齐到地图的lio
- /as_lio/lmap_state: std_msgs::String: 重定位节点状态
#### service
- /as_lio/ctrl: mower_msgs::Trigger: as_lio节点控制
  - "reset_lio": lio归零
  - "start_mapping": 开始建图
  - "stop_mapping": 停止建图
  - "open_lio": 启用lio
  - "close_lio": 关闭lio
- /as_lio/savemap: mower_msgs::Trigger: 保存地图, 同时结束建图
- /as_lio/loadmap: mower_msgs::Trigger: 加载地图

### Vision/Lidar 工作逻辑
#### 建图
1. 机器停在充电桩上, APP进入建图模式, 下发建图指令
   - fusion节点受到开始建图指令, 以在桩位置为原点进行初始化
   - fusion节点下发指令到 视觉/激光 节点, 重置状态, 以当前位置为原点进行建图, 进入建图模式
2. 人工遥控机器进行建图，圈出边界区域
   - 视觉/激光 节点持续发布里程计
3. 结束建图, 点击APP保存地图, 下发保存地图指令
   - fusion节点收到保存地图指令, 将保存指令下发到 视觉/激光 节点
   - 视觉/激光 节点保存地图，并退出建图模式

#### 定位
1. APP端选择地图，加载, 下发任务指令
   - fusion节点收到加载地图指令, 将加载指令下发到 视觉/激光 节点
   - 视觉/激光 节点加载地图，并进入定位模式
2. 机器开始运动, 视觉/激光 节点持续发布重定位里程计

### RTK-Vision 工作逻辑
#### 建图
1. 机器停在充电桩上, APP进入建图模式, 下发建图指令
   - fusion节点受到开始建图指令, 以在桩位置为原点进行初始化: 同步进行gnss-initor 和 vio-gnss-initor
   - 1. gnss-initor: 优先使用gnss初始化, 如果gnss初始化失败, 则使用vio-gnss-initor; 如果gnss初始化成功, 则vio-gnss-initor不执行
   - gnss-offset计算完后, vio-tracker启动, 进入rtk-vio融合跟踪模式
#### 定位
1. 桩上启动: 直接初始化
2. 桩外启动: gnss初始化, 失败后直接报错