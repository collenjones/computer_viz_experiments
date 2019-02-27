#include <algorithm>
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
      // These floats represent the 2x2 H matrix
      //   Ix^2  Ixy
      //   Ixy   Iy^2
      float a11, a12,
            a21, a22;
      
      a11 = derivatives.Ix.at<float>(row, col) * derivatives.Ix.at<float>(row, col);
      a12 = derivatives.Ix.at<float>(row, col) * derivatives.Iy.at<float>(row, col);
      a21 = a12;
      a22 = derivatives.Iy.at<float>(row, col) * derivatives.Iy.at<float>(row, col);
      
      float det = a11 * a22 - a12 * a21;
      float trace = a11 + a22;
      
      double R = abs(det - (k * trace * trace));
      interest_points.at<float>(row, col) = R;
    }
  }
  return interest_points;
}

std::vector<harris::InterestPoint> get_top_tile_interest_points(const cv::Mat &tile_window, unsigned int top_left_x, unsigned int top_left_y, unsigned int num_per_tile) {
  std::vector<harris::InterestPoint> window_interest_points;
  for (int i = 0; i < tile_window.cols; ++i) {
    for (int j = 0; j < tile_window.rows; ++j) {
      harris::InterestPoint interest_point;
      interest_point.corner_value = tile_window.at<float>(j, i);
      if (interest_point.corner_value < 10000) {
        continue;
      }
      interest_point.point = cv::Point(top_left_x + i, top_left_y + j);
      window_interest_points.emplace_back(interest_point);
    }
  }
  if (window_interest_points.size() > 0) {
    std::sort(window_interest_points.begin(), window_interest_points.end(), [](harris::InterestPoint i1, harris::InterestPoint i2) { return i1.corner_value > i2.corner_value; });
  }
  return window_interest_points;
}

std::vector<harris::InterestPoint> harris::suppress_nonmax(const cv::Mat &interest_points, unsigned int num_per_tile) {
  std::vector<harris::InterestPoint> interest_point_maximas;
  
  int window_width = interest_points.cols / 10;
  int window_height = interest_points.rows / 10;
  
  for (int height = 0; height < interest_points.rows; height += window_height) {
    if (interest_points.rows - height < window_height) {
      break;
    }
    for (int width = 0; width < interest_points.cols; width += window_width) {
      if (interest_points.cols - width < window_width) {
        break;
      }
      cv::Mat window = interest_points(cv::Range(height, height + window_height), cv::Range(width, width + window_width));
      std::vector<harris::InterestPoint> window_interest_points = get_top_tile_interest_points(window, width, height, num_per_tile);
      if (window_interest_points.size() > 0) {
        std::copy(window_interest_points.begin(), window_interest_points.begin() + num_per_tile, std::back_inserter(interest_point_maximas));
      }
    }
  }
  
  return interest_point_maximas;
}
