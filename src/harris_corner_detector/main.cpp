#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "harris.hpp"

// TODO:
//   Don't use arbitrary k value
//   Improve thresholding
//   Profile & improve efficiency

//Global variables
std::vector<harris::InterestPoint> interest_point_maximas;
const float k = 0.04;
const unsigned int min_pixel_radius = 10;

cv::Mat highlight_features(const cv::Mat &image, const std::vector<harris::InterestPoint> &interest_point_maximas) {
  cv::Mat new_image(image);
  for(auto interest_point : interest_point_maximas) {
    // Draw a red circle with a black outline
    cv::circle(new_image, interest_point.point, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    cv::circle(new_image, interest_point.point, 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
  return new_image;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
  if ( event == cv::EVENT_MOUSEMOVE ) {
    for (auto interest_point : interest_point_maximas) {
      if ((abs(x - interest_point.point.x) < min_pixel_radius) && (abs(y - interest_point.point.y) < min_pixel_radius)) {
        std::cout << "Interest Point corner value: " << interest_point.corner_value << std::endl;
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: harris_corner_detector <filename>" << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat scaled_down_image;
  if (!image.data) {
    std::cout << "Could not open file or find the image: " << argv[1] << std::endl;
    return -1;
  }

  // Reduce image size to reduce number of calculations required
  double scale_factor = 0.5;
  cv::resize(image, scaled_down_image, cv::Size(image.cols * scale_factor, image.rows * scale_factor));
  
  std::cout << "Getting interest points..." << std::endl;
  cv::Mat interest_points = harris::get_interest_points(scaled_down_image, 7, k);
  interest_point_maximas = harris::suppress_nonmax(interest_points, 10, min_pixel_radius);
  
  std::cout << "Drawing interest points..." << std::endl;
  std::string window_name("Interest Points");
  cv::namedWindow(window_name, 1);
  // set the callback function for any mouse event
  cv::setMouseCallback(window_name, CallBackFunc, NULL);
  cv::imshow(window_name, highlight_features(scaled_down_image, interest_point_maximas));
  
  std::cout << "Interest points ready..." << std::endl;
  cv::waitKey(0);
  return 0;
}
