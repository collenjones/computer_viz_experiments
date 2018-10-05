#include <algorithm>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// TODO: Can we do better than hardcoding this value?
const uint FEATURE_THRESHOLD = 10000;

double get_euclidean_distance(int x1, int y1, int x2, int y2) {
  return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

cv::SparseMat suppress_nonmax(const cv::SparseMat &interest_points, const cv::Size &window_size, int max_per_window=5, uint min_pixel_distance=40) {
  cv::SparseMat salient_points(2, interest_points.size(), CV_32F);
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

cv::SparseMat get_interest_points(const cv::Mat &image, uint descriptor_image_width) {
  cv::SparseMat interest_points(2, (int[]){image.rows, image.cols}, CV_32F);
  cv::Mat image_gray;
  cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  
  cv::Mat grad_x, grad_y, Ix2, Iy2, Ixy;
  // Step 1. Calculate Gaussian image gradients in x and y direction
  cv::Sobel(image_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(image_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
  // Step 2. Compute three images (Ix2, Iy2, Ixy) from the outer products of the gradients
  Ix2 = grad_x.mul(grad_x);
  Iy2 = grad_y.mul(grad_y); 
  Ixy = grad_x.mul(grad_y);
  
  const int window_offset = descriptor_image_width / 2;
  for (int row = window_offset; row < image_gray.rows - window_offset; row++) {
    for (int col = window_offset; col < image_gray.cols - window_offset; col++) {
      cv::Mat Ix2_window = Ix2.rowRange(row - window_offset, row + window_offset + 1)
                              .colRange(col - window_offset, col + window_offset + 1);
      cv::Mat Iy2_window = Iy2.rowRange(row - window_offset, row + window_offset + 1)
                              .colRange(col - window_offset, col + window_offset + 1);
      cv::Mat Ixy_window = Ixy.rowRange(row - window_offset, row + window_offset + 1)
                              .colRange(col - window_offset, col + window_offset + 1);
      
      double Sx2 = cv::sum(Ix2_window)[0];
      double Sy2 = cv::sum(Iy2_window)[0];
      double Sxy = cv::sum(Ixy_window)[0];
      cv::Mat H = (cv::Mat_<double>(2, 2) << Sx2, Sxy, Sxy, Sy2);
      double trace = cv::trace(H)[0];
      // Step 3. Compute corner response function R
      double R = cv::determinant(H) - (0.04 * trace * trace);
      // Step 4. Threshold R
      if (R > FEATURE_THRESHOLD) {
        interest_points.ref<float>(row, col) = R;
      }
    }
  }

  // Step 5. TBD (non-max suppression)
  return suppress_nonmax(interest_points, cv::Size(48, 48));
}

cv::Mat highlight_features(const cv::Mat &image, const cv::SparseMat &interest_points) {
  cv::Mat new_image(image);
  cv::SparseMatConstIterator_<float> it = interest_points.begin<float>();
  cv::SparseMatConstIterator_<float> it_end = interest_points.end<float>();
  for(; it != it_end; ++it) {
    const cv::SparseMat::Node *node = it.node();
    cv::circle(new_image, cv::Point(node->idx[1], node->idx[0]), 3, cv::Scalar(0, 0, 255));
  }
  return new_image;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: harris_corner_detector <filename>" << std::endl;
    return -1;
  }
  
  cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (!image.data) {
    std::cout << "Could not open file or find the image: " << argv[1] << std::endl;
    return -1;
  }
  
  cv::SparseMat interest_points = get_interest_points(image, 5);
  
  // TODO: Implement non-max suppression
  cv::imshow("Features", highlight_features(image, interest_points));
  cv::waitKey(0);
  return 0;
}
