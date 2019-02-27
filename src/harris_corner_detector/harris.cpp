#include <algorithm>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "harris.hpp"

const int CORNER_DETECTION_THRESHOLD = 10000;

struct Derivatives {
  cv::Mat Ix2;
  cv::Mat Iy2;
  cv::Mat Ixy;
};

Derivatives get_derivatives(const cv::Mat &image) {
  cv::Mat image_gray;
  cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  
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
  d.Ix2 = Ix.mul(Ix);
  d.Iy2 = Iy.mul(Iy);
  d.Ixy = Ixy;
  
  return d;
}

double compute_corner_value(const cv::Mat &image, const cv::Rect &rect, const Derivatives &d, float k) {
  cv::Mat sliding_window, sIx2, sIy2, sIxy;
  sliding_window = image(rect);
  sIx2 = d.Ix2(rect);
  sIy2 = d.Iy2(rect);
  sIxy = d.Ixy(rect);
  double a11, a12,
         a21, a22;
  a11 = cv::sum(sIx2)[0];
  a12 = cv::sum(sIxy)[0];
  a21 = a12;
  a22 = cv::sum(sIy2)[0];
  
  double determinant = a11 * a22 - a12 * a21;
  double trace = a11 + a22;
  
  return determinant - (k * trace * trace);
}

cv::Mat harris::get_interest_points(const cv::Mat &image, unsigned int kernel_size, float k) {
  Derivatives derivatives = get_derivatives(image);
  cv::Mat interest_points(derivatives.Ix2.rows, derivatives.Ix2.cols, CV_32F);
  
  for (int row = 0; row < derivatives.Iy2.rows - kernel_size; ++row) {
    for (int col = 0; col < derivatives.Iy2.cols - kernel_size; ++col) {
      double corner_value = compute_corner_value(image, cv::Rect(col, row, kernel_size, kernel_size), derivatives, k);
      interest_points.at<float>(row + (kernel_size / 2), col + (kernel_size / 2)) = corner_value;
    }
  }
  return interest_points;
}

std::vector<harris::InterestPoint> get_top_tile_interest_points(const cv::Mat &tile_window, unsigned int top_left_x, unsigned int top_left_y, unsigned int num_per_tile) {
  std::vector<harris::InterestPoint> window_interest_points;
  for (int c = 0; c < tile_window.cols; ++c) {
    for (int r = 0; r < tile_window.rows; ++r) {
      harris::InterestPoint interest_point;
      interest_point.corner_value = tile_window.at<float>(r, c);
      if (interest_point.corner_value < CORNER_DETECTION_THRESHOLD) {
        continue;
      }
      interest_point.point = cv::Point(static_cast<int>(top_left_x) + c, static_cast<int>(top_left_y) + r);
      window_interest_points.emplace_back(interest_point);
    }
  }
  if (window_interest_points.size() == 0) {
    return window_interest_points;
  }
  std::sort(window_interest_points.begin(), window_interest_points.end(), [](harris::InterestPoint i1, harris::InterestPoint i2) { return i1.corner_value > i2.corner_value; });
  std::vector<harris::InterestPoint> trimmed_points(window_interest_points.begin(), std::min(window_interest_points.begin() + num_per_tile, window_interest_points.end()));
  return trimmed_points;
}

std::vector<harris::InterestPoint> harris::suppress_nonmax(const cv::Mat &interest_points, unsigned int num_per_tile, unsigned min_pixel_radius) {
  std::vector<harris::InterestPoint> interest_point_maximas;
  cv::Mat suppression_matrix(interest_points.rows, interest_points.cols, CV_32F, cv::Scalar::all(0));
  
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

      // Enforce minimum pixel radius between interest points
      for (int i = 0; i < window_interest_points.size(); ++i) {
        harris:InterestPoint ip = window_interest_points[i];
        if (suppression_matrix.at<int>(ip.point) == 0) {
          for (int r = -min_pixel_radius; r <= static_cast<int>(min_pixel_radius); ++r) {
            for (int c = -min_pixel_radius; c <= static_cast<int>(min_pixel_radius); ++c) {
              int sr = ip.point.y + r;
              int sc = ip.point.x + c;
              
              // bounds checking
              if (sr >= suppression_matrix.rows)
                sr = suppression_matrix.rows - 1;
              if (sr < 0)
                sr = 0;
              if (sc >= suppression_matrix.cols)
                sc = suppression_matrix.cols - 1;
              if (sc < 0)
                sc = 0;
              
              suppression_matrix.at<int>(sr, sc) = 1;
            }
          }
          interest_point_maximas.emplace_back(ip);
        }
      }
    }
  }

  return interest_point_maximas;
}
