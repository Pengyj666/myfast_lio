#ifndef ICP_3D_H
#define ICP_3D_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <limits>
#include <vector>
#include <pcl/features/normal_3d.h>
#include "ikd-Tree/ikd_Tree.h"

// 定义点类型和相关类型别名
using PointT = pcl::PointXYZINormal;
typedef std::vector<PointT, Eigen::aligned_allocator<PointT>> PointVector;

// 参数宏定义
#ifndef NUM_MATCH_POINTS
#define NUM_MATCH_POINTS 5
#endif


class Icp3d {
public:
    Icp3d() = default;
    virtual ~Icp3d();

    // 点云预处理函数
    pcl::PointCloud<PointT>::Ptr preprocessCloud(
        const pcl::PointCloud<PointT>::Ptr& cloud,
        float voxel_size);

    // 雅可比矩阵计算函数
    Eigen::Matrix<double, 1, 6> computeJacobianCentralDifference(
        const PointT& point,
        const Eigen::Matrix<float, 4, 1>& plane_params,
        const Eigen::Matrix4f& transformation,
        double step_size = 1e-6);

    // 点到平面ICP算法主函数
    std::pair<Eigen::Matrix4f, double> pointToPlaneICP(
        const pcl::PointCloud<PointT>::Ptr& source,
        const pcl::PointCloud<PointT>::Ptr& target,
        const Eigen::Matrix4f& initial,
        int max_iterations = 30,
        double source_size = 0.1, // 默认SCAN_VOXEL_SIZE
        double transformation_epsilon = 1e-6);

    // KD树构建函数
    void kdtree_bulid(int scale, pcl::PointCloud<pcl::PointXYZINormal>::Ptr& map_cloud);

protected:
    double MAP_VOXEL_SIZE;
    KD_TREE<PointT> kdtree;
    // 平面估计模板函数声明
    template<typename T>
    bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold);
};

template<typename T>
bool Icp3d::esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    // 构建线性方程组的系数矩阵A和常数向量b
    // 方程形式为: ax + by + cz = -1
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    // 使用QR分解求解线性方程组，得到平面的法向量
    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    // 归一化法向量并计算平面参数
    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    // 检查所有点到拟合平面的距离是否都在阈值范围内
    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

#endif // ICP_3D_H