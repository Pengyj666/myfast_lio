#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <fstream>

#include "common/sysutils.h"

bool getNodePid(const std::string& node_name, pid_t& pid) {
  std::string command = "rosnode info " + node_name + " | grep Pid | awk '{print $2}'";
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return false;
  
  char buffer[128];
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    pid = static_cast<pid_t>(atoi(buffer));
    pclose(pipe);
    return true;
  }
  
  pclose(pipe);
  return false;
}

bool getProcPid(const std::string& proc_name, pid_t& pid) {
  std::string command = "pidof " + proc_name;
  std::string pid_str = utils::ExecWithStdout(command);
  if (pid_str.size() > 0) {
    if (std::all_of(pid_str.begin(), pid_str.end(), [](unsigned char c) { return std::isdigit(c) || c == '\n'; } )) {
      pid = static_cast<pid_t>(std::atoi(pid_str.c_str()));
      return true;
    } else {
      std::printf("getProcPid(): call cmd '%s' ret bad: %s!\n", command.c_str(), pid_str.c_str());
    }
  } else {
    std::printf("getProcPid(): call cmd '%s' Failed!\n", command.c_str());
  }
  return false;
}

// 单位: 字节
// 总程序大小:  mem_info[0]
// 驻留集大小(RSS):  mem_info[1]
// 共享页面:  mem_info[2]
// 文本(代码):  mem_info[3]
// 数据/栈:  mem_info[4]
std::vector<size_t> getProcessMemoryUsage(pid_t pid) {
  std::vector<size_t> memory_info;
  std::string path = "/proc/" + std::to_string(pid) + "/statm";
  std::ifstream statm_file(path);
  
  if (statm_file.is_open()) {
      std::string line;
      std::getline(statm_file, line);
      std::istringstream iss(line);
      
      size_t value;
      while (iss >> value) {
          memory_info.push_back(value * sysconf(_SC_PAGESIZE)); // 转换为字节
      }
  }
  
  return memory_info;
}

void test_mem() {
  float mb = 1024 * 1024;
  while (true) {
    pid_t pid;
    auto ts1 = utils::GetNow_Steady();
    if (getNodePid("as_vio", pid)) {
      auto ts2 = utils::GetNow_Steady();
      auto mem_info = getProcessMemoryUsage(pid);
      auto ts3 = utils::GetNow_Steady();
      
      std::cout << "耗时: get_pid=" << ts2-ts1 << " ms, get_mem=" << ts3-ts2 << " ms" << std::endl;
      if (!mem_info.empty()) {
        std::cout << std::endl;
        std::cout << "内存信息 (单位: MB):" << std::endl;
        std::cout << "总程序大小: " << mem_info[0] / mb << std::endl;
        std::cout << "驻留集大小(RSS): " << mem_info[1] / mb << std::endl;
        std::cout << "共享页面: " << mem_info[2] / mb << std::endl;
        std::cout << "文本(代码): " << mem_info[3] / mb << std::endl;
        std::cout << "数据/栈: " << mem_info[4] / mb << std::endl;
      } else {
        std::cerr << "无法获取内存信息" << std::endl;
      }
    }
    utils::Sleep(1000);
  }
}

void test_tf() {
  // 左相机在右相机坐标系下的旋转和平移，相机front-Z, down-Y, right-X
  // sn106机器
  // Eigen::Matrix3d R_rl; 
  // R_rl << 0.9999329847280656, -0.003418232257114446, 0.01106082008978512,
  //         0.00339167769761287, 0.9999913233022771, 0.002418644776751328,
  //         -0.01106899160798703, -0.002380967953798056, 0.9999359021539257;
  // Eigen::Vector3d t_rl;
  // t_rl << -0.05991530990600586, 0.00008932638168334962, 0.0006033118367195129;

  // sn108机器
  // Eigen::Matrix3d R_rl; 
  // R_rl << 0.9999894158974774, -0.0008726008391552204, 0.004517373218768649,
  //         0.0008639564200480427, 0.9999977928069066, 0.001915192058077659,
  //         -0.004519034446250702, -0.001911268973894458, 0.9999879626168422;
  // Eigen::Vector3d t_rl;
  // t_rl << -0.06005000305175781, 0.00004389339312911034, 0.0003248438239097595;

//   // sn201机器
//   Eigen::Matrix3d R_rl; 
//   R_rl << 0.9999883205786456, 0.003419805517172745, 0.003415206659145317,
//           -0.003416013628679896, 0.9999935432208872, -0.001115512180309636,
//           -0.003418999442618976, 0.001103832759280501, 0.9999935459771983;
//   Eigen::Vector3d t_rl;
//   t_rl << -0.05994360733032227, 0.000411138653755188, 0.0002553612291812897;

  // sn201_new机器
  Eigen::Matrix3d R_rl; 
  R_rl << 0.9999131, -0.0095055, -0.0091339,
          0.0095426,  0.9999464,  0.0040308,
          0.0090951, -0.0041176,  0.9999502;
  Eigen::Vector3d t_rl;
  t_rl << -0.060093117, -0.000378853, -0.000226824;

  // sn210机器
//   Eigen::Matrix3d R_rl; 
//   R_rl << 0.9999883205786456, 0.003419805517172745, 0.003415206659145317,
//           -0.003416013628679896, 0.9999935432208872, -0.001115512180309636,
//           -0.003418999442618976, 0.001103832759280501, 0.9999935459771983;
//   Eigen::Vector3d t_rl;
//   t_rl << -0.05994360733032227, 0.000411138653755188, 0.0002553612291812897;

  // sn233机器
  // Eigen::Matrix3d R_rl; 
  // R_rl << 0.9999855548219636, 0.004477817640908698, 0.002973095454987287,
  //         -0.00446294429634906, 0.9999775878945226, -0.004990574802564366,
  //         -0.002995375705547568, 0.004977233953419493, 0.9999831272909336;
  // Eigen::Vector3d t_rl;
  // t_rl << -0.06000407028198242, 0.00005536968261003494, -0.00004870776832103729;

  // sn235机器
  // Eigen::Matrix3d R_rl; 
  // R_rl << 0.9999936924505154, -0.00156374020636017, 0.003189008584344057,
  //         0.001588999140753047, 0.9999672665484268, -0.007933530953996569,
  //         -0.003176498215754814, 0.007938548244757872, 0.9999634439872545;
  // Eigen::Vector3d t_rl;
  // t_rl << -0.06006144714355469, -0.000263557106256485, 0.0001441267132759094;

  // sn241机器
//   Eigen::Matrix3d R_rl; 
//   R_rl << 0.9999912845676348, -0.0003638477922453798, 0.00415913495282183,
//           0.0003648506847415778, 0.999999904552184, -0.0002403738769113427,
//           -0.004159047096337055, 0.0002418892451845481, 0.9999913218707668;
//   Eigen::Vector3d t_rl;
//   t_rl << -0.0599907112121582, -0.00008346114307641983, 0.0001592071801424026;

  // 左相机在imu坐标系下的旋转和平移，imu(以相机机身看) back-Z, down-Y, left-Z
  Eigen::Matrix3d R_bl;
  R_bl << -1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, -1.0;
  Eigen::Vector3d t_bl;
  t_bl << 0.05143, -0.00453, -0.01503;

  Eigen::Matrix3d R_br;
  Eigen::Vector3d t_br;

  // T_bl = T_br * T_rl
  // T_br = T_bl * T_rl.inverse()
  //
  // (R_br|t_br) = (R_bl|t_bl) * (R_rl|t_rl).inverse()
  //             = (R_bl|t_bl) * (R_rl.inv()| -R_rl.inv()*t_rl)
  //             = (R_bl * R_rl.inv()) | (R_bl * (-R_rl.inv()*t_rl) + t_bl)
  //
  // (R1|t1) * (R2|t2) = ((R1 * R2) | (R1 * t2 + t1))
  R_br = R_bl * R_rl.inverse();
  t_br = R_bl * (-R_rl.inverse() * t_rl) + t_bl;

  std::cout << "R_br = \n" << R_br << std::endl;
  std::cout << "t_br = \n" << t_br << std::endl;
}

void test_slerp() {
  Eigen::Quaterniond q1(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()));
  Eigen::Quaterniond q2(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()));
  Eigen::Quaterniond q11 = q1.slerp(0.1, q2); // （1.0-0.1）*q1 + 0.1*q2
  Eigen::Quaterniond q12 = q1.slerp(0.5, q2); // （1.0-0.5）*q1 + 0.5*q2
  Eigen::Quaterniond q13 = q1.slerp(0.9, q2); // （1.0-0.9）*q1 + 0.9*q2

  std::cout << "q1 = " << q1.coeffs() << std::endl;
  std::cout << "q2 = " << q2.coeffs() << std::endl;
  std::cout << "q11 = " << q11.coeffs() << std::endl;
  std::cout << "q12 = " << q12.coeffs() << std::endl;
  std::cout << "q13 = " << q13.coeffs() << std::endl;
}

int main(int argc, char **argv) {
  // test_tf();
  // test_mem();
  test_slerp();
  return 0;
}