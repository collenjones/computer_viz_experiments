#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "harris.hpp"

struct Derivatives {
  cv::Mat Ix;
  cv::Mat Iy;
  cv::Mat Ixy;
};

Derivatives get_derivatives(const cv::Mat &image) {
  cv::Mat image_gray;
  cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(image_gray, image_gray, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT);
  
  cv::Mat Ix, Iy, Ixy;
  cv::Sobel(image_gray, Ix, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(image_gray, Iy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  Ixy = Ix.mul(Iy);
  
  // Apply secondary Gaussian blur with larger kernel
  cv::Size derivative_blur_kernel = cv::Size(7,7);
  cv::GaussianBlur(Ix, Ix, derivative_blur_kernel, 0, 0, cv::BORDER_DEFAULT);
  cv::GaussianBlur(Iy, Iy, derivative_blur_kernel, 0, 0, cv::BORDER_DEFAULT);
  cv::GaussianBlur(Ixy, Ixy, derivative_blur_kernel, 0, 0, cv::BORDER_DEFAULT);
  
  Derivatives d;
  d.Ix = Ix;
  d.Iy = Iy;
  d.Ixy = Iy;
  
  return d;
}

cv::Mat harris::get_interest_points(const cv::Mat &image, float k) {
  Derivatives derivatives = get_derivatives(image);
  cv::Mat interest_points(derivatives.Ix.rows, derivatives.Ix.cols, CV_32F);
  
  for (int row = 0; row < derivatives.Iy.rows; ++row) {
    for (int col = 0; col < derivatives.Iy.cols; ++col) {
      float Ix = derivatives.Ix.at<float>(row, col);
      float Iy = derivatives.Iy.at<float>(row, col);
      float Ixy = Ix * Iy;
      // TODO: Remove this Mat -- will be faster to calculate determinant and trace directly
      cv::Mat H = (cv::Mat_<double>(2, 2) << Ix * Ix, Ixy,
                                             Ixy    , Iy * Iy);
      double trace = cv::trace(H)[0];
      double R = abs(cv::determinant(H) - (k * trace * trace));
      interest_points.at<float>(row, col) = R;
    }
  }
  return interest_points;
}

std::vector<harris::InterestPoint> harris::suppress_nonmax(const cv::Mat &interest_points) {
  std::vector<harris::InterestPoint> interest_point_maximas;
  
  for (int row = 0; row < interest_points.rows; ++row) {
    for (int col = 0; col < interest_points.cols; ++col) {
      float corner_value = interest_points.at<float>(row, col);
      if (corner_value > 10000) {
        cv::Point point(col, row);
        harris::InterestPoint interest_point;
        interest_point.point = point;
        interest_point.corner_value = corner_value;
        interest_point_maximas.emplace_back(interest_point);
      }
    }
  }
  
  return interest_point_maximas;
}
