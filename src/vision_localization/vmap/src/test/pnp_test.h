#include <opencv2/opencv.hpp>

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

const cv::Mat Rm_rdf2flu = (cv::Mat_<float>(3, 3) << 0,  0, 1, 
                                                    -1,  0, 0, 
                                                     0, -1, 0);
const cv::Mat Rm_flu2rdf = (cv::Mat_<float>(3, 3) << 0, -1, 0, 
                                                     0, 0, -1, 
                                                     1, 0, 0);

cv::Point3f img2cam(cv::Point2f img_point, float depth, cv::Mat camera_matrix) {
  float fx = camera_matrix.at<float>(0, 0);
  float fy = camera_matrix.at<float>(1, 1);
  float cx = camera_matrix.at<float>(0, 2);
  float cy = camera_matrix.at<float>(1, 2);
  return cv::Point3f((img_point.x - cx) * depth / fx, (img_point.y - cy) * depth / fy, depth);
}

cv::Point2f cam2img(cv::Point3f cam_point, cv::Mat camera_matrix) {
  float fx = camera_matrix.at<float>(0, 0);
  float fy = camera_matrix.at<float>(1, 1);
  float cx = camera_matrix.at<float>(0, 2);
  float cy = camera_matrix.at<float>(1, 2);
  return cv::Point2f(cx + cam_point.x * fx / cam_point.z, cy + cam_point.y * fy / cam_point.z);
}

// 相机坐标系rdf: right-X, down-Y, front-Z
// 相机坐标系flu: front-X, left-Y, up-Z
cv::Point3f cam_rdf2flu(cv::Point3f cam_point) {
  return cv::Point3f(cam_point.z, -cam_point.x, -cam_point.y);
}

cv::Point3f cam_flu2rdf(cv::Point3f cam_point) {
  return cv::Point3f(-cam_point.y, -cam_point.z, cam_point.x);
}

int pnp_test() {
  int rows = 480, cols = 640;
  float fx = 450.0, fy = 450.0;
  float cx = 320.0, cy = 240.0;

  cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat dist_coeffs = (cv::Mat_<float>(1, 5) << 0, 0, 0, 0, 0);
  cv::Mat rvec, tvec;

  // 2D points in the image coordinate system
  std::vector<cv::Point2f> image_points;
  image_points.push_back(cv::Point2f(cx-200.0, cy-200.0));
  image_points.push_back(cv::Point2f(cx-200.0, cy+200.0));
  image_points.push_back(cv::Point2f(cx+200.0, cy-200.0));
  image_points.push_back(cv::Point2f(cx+200.0, cy+200.0));
  // 3D points in the object coordinate system
  std::vector<cv::Point3f> object_points;
  for (int i = 0; i < image_points.size(); i++) {
    object_points.push_back(cam_rdf2flu(img2cam(image_points[i], 1.0, camera_matrix)));
  }
  
  std::vector<cv::Point2f> image_points2;
  for (int i = 0; i < object_points.size(); i++) {
    image_points2.push_back(cam2img(cam_flu2rdf(object_points[i]), camera_matrix));
  }

  for (int i = 0; i < object_points.size(); i++) {
    std::printf("imgP=(%.1f,%.1f), imgP2=(%.1f,%.1f), objP=(%.2f,%.2f,%.2f)\n", 
        image_points[i].x, image_points[i].y, image_points2[i].x, image_points2[i].y,
        object_points[i].x, object_points[i].y, object_points[i].z);
  }

  cv::Mat R = (cv::Mat_<float>(3, 3) << 0.f, -1.f, 0.f, 
                                        1.f,  0.f, 0.f, 
                                        0.f,  0.f, 1.f);
  cv::Mat t = (cv::Mat_<float>(3, 1) << 1.0, 1.0, 3.0);

  std::printf("\n");
  std::cout << "R: " << R << std::endl;
  std::cout << "t: " << t << std::endl;

  for (int i = 0; i < object_points.size(); i++) {
    cv::Mat pt = (cv::Mat_<float>(3, 1) << object_points[i].x, object_points[i].y, object_points[i].z);
    pt = R * pt + t;
    object_points[i] = cv::Point3f(pt.at<float>(0, 0), pt.at<float>(1, 0), pt.at<float>(2, 0));
  }
  std::printf("\n");
  for (int i = 0; i < object_points.size(); i++) {
    std::printf("nav_objP=(%.2f,%.2f,%.2f)\n", object_points[i].x, object_points[i].y, object_points[i].z);
  }
  std::printf("\n");

  // 点云先从世界坐标系flu转到相机坐标系rdf
  for (int i = 0; i < object_points.size(); i++) {
    object_points[i] = cam_flu2rdf(object_points[i]);
  }

  // object_points 是在 cam_rdf 坐标系下
  // 计算得到的rvec 和 tvec 也是在 cam_rdf 坐标系下
  // 且满足: object_points = R * cam_rdf + tvec
  cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

  cv::Mat pnp_R;
  cv::Rodrigues(rvec, pnp_R);
  std::cout << "pnp_R: " << pnp_R << std::endl;
  std::cout << "pnp_t: " << tvec << std::endl;

  cv::Mat reloc_Rcw = (cv::Mat_<float>(3, 3) << pnp_R.at<double>(0, 0), pnp_R.at<double>(0, 1), pnp_R.at<double>(0, 2),
                                              pnp_R.at<double>(1, 0), pnp_R.at<double>(1, 1), pnp_R.at<double>(1, 2),
                                              pnp_R.at<double>(2, 0), pnp_R.at<double>(2, 1), pnp_R.at<double>(2, 2));
  cv::Mat reloc_tcw = (cv::Mat_<float>(3, 1) << tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));

  std::printf("\n");
  std::cout << "reloc_R: " << reloc_Rcw << std::endl;
  std::cout << "reloc_t: " << reloc_tcw << std::endl;

  cv::Mat reloc_Rwc = reloc_Rcw.t();
  cv::Mat reloc_twc = -reloc_Rwc * reloc_tcw;  

  reloc_Rwc = Rm_rdf2flu * reloc_Rwc * Rm_rdf2flu.t();
  reloc_twc = Rm_rdf2flu * reloc_twc;

  std::printf("\n");
  std::cout << "R_reloc: " << reloc_Rwc << std::endl;
  std::cout << "t_reloc: " << reloc_twc << std::endl;
  return 0;
}