#include <opencv2/imgproc/imgproc.hpp>
#include "harris.hpp"

const uint FEATURE_THRESHOLD = 10000;

struct Derivatives {
  cv::Mat Ix;
  cv::Mat Iy;
  cv::Mat Ixy;
}

cv::Mat suppress_nonmax(const cv::Mat &interest_points, const cv::Size &window_size, int max_per_window=5, uint min_pixel_distance=10) {
  cv::Mat salient_points(2, interest_points.size(), CV_32F);
  for (int x = 0; x < interest_points.size()[1]; x += window_size.width) {
    for (int y = 0; y < interest_points.size()[0]; y += window_size.height) {
      std::vector<std::tuple<int, int, float>> window_points;  // x, y, R
      cv::SparseMatConstIterator_<float> it = interest_points.begin<float>();
      cv::SparseMatConstIterator_<float> it_end = interest_points.end<float>();
      // Get points in window
      for(; it != it_end; ++it) {
        const cv::SparseMat::Node *node = it.node();
        int point_x = node->idx[1];
        int point_y = node->idx[0];
        bool is_point_inside_window = (point_x >= x && point_x <= x + window_size.width) && (point_y >= y && point_y <= y + window_size.height);
        if (is_point_inside_window) {
          window_points.push_back(std::make_tuple(point_y, point_x, *it));
        }
      }
      
      // Sort by R (largest first)
      struct {
        bool operator()(std::tuple<int, int, float> a, std::tuple<int, int, float> b) const
        {
          return std::get<2>(a) > std::get<2>(b);
        }
      } customGreater;
      std::sort(std::begin(window_points), std::end(window_points), customGreater);
      
      if (window_points.size() == 0) {
        continue;
      }
      salient_points.ref<float>(std::get<0>(window_points.at(0)), std::get<1>(window_points.at(0))) = std::get<2>(window_points.at(0));
      int points_taken = 1;
      int last_good_point_index = 0;
      for (int i = 1; i < (int)window_points.size(); i++) {
        std::tuple<int, int, float> point = window_points.at(i);
        std::tuple<int, int, float> last_good_point = window_points.at(last_good_point_index);
        // Ensure points are not too close to each other
        bool is_within_range = get_euclidean_distance(std::get<0>(point), std::get<1>(point),
                                                      std::get<0>(last_good_point), std::get<1>(last_good_point)) >= min_pixel_distance;
        bool can_still_accept_points = points_taken < max_per_window;
        if (can_still_accept_points && is_within_range) {
          salient_points.ref<float>(std::get<0>(point), std::get<1>(point)) = std::get<2>(point);
          points_taken++;
          last_good_point_index = i;
        }
      }
    }
  }
  return salient_points;
}

Derivatives get_derivatives(const cv::Mat &image) {
  cv::Mat image_gray;
  cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(image_gray, image_gray, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT);
  
  cv::Mat Ix, Iy, Ixy;
  cv::Sobel(image_gray, Ix, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(image_gray, Iy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
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

cv::Mat harris::get_interest_points(const cv::Mat &image, float k, uint descriptor_image_width) {
  Derivatives derivatives = get_derivatives(image);
  cv::Mat interest_points(derivatives.Ix.rows, derivatives.Ix.cols, CV_32F);
  
  for (int row = 0; row < derivatives.Ix.rows; ++row) {
    for (int col = 0; col < derivatives.Ix.cols; ++col) {
      float Ix = derivatives.Ix.at<float>(row, col);
      float Iy = derivatives.Iy.at<float>(row, col);
      float Ixy = derivatives.Ix.at<float>(row, col) * derivatives.Iy.at<float>(row, col);
      cv::Mat H = (cv::Mat_<double>(2, 2) << Ix * Ix, Ixy,
                                             Ixy    , Iy * Iy);
      double trace = cv::trace(H)[0];
      double R = abs(cv::determinant(H) - (k * trace * trace));
      interest_points.at<float>(row, col) = R;
    }
  }
  return interest_points;
}
