#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// TODO: Can we do better than hardcoding this value?
const uint FEATURE_THRESHOLD = 10000;

std::vector<cv::Point> get_interest_points(const cv::Mat &image, uint descriptor_image_width) {
  cv::Mat image_gray;
  cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  std::vector<cv::Point> interest_points;
  
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
        interest_points.push_back(cv::Point(col, row));
      }
    }
  }

  // Step 5. TBD (non-max suppression)
  return interest_points;
}

cv::Mat highlight_features(const cv::Mat &image, const std::vector<cv::Point> &interest_points) {
  cv::Mat new_image(image);
  for (auto point : interest_points) {
    cv::circle(new_image, point, 5, cv::Scalar(0, 0, 255));
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
  
  std::vector<cv::Point> interest_points = get_interest_points(image, 5);
  
  // TODO: Implement non-max suppression
  cv::imshow("Features", highlight_features(image, interest_points));
  cv::waitKey(0);
  return 0;
}
