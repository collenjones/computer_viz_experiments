#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: gaussian_noise <path to image>" << std::endl;
    return -1;
  }
  
  cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (!image.data) {
    std::cout << "Could not open file or find the image: " << argv[1] << std::endl;
    return -1;
  }
  
  cv::Mat noise(image.size(), image.type());
  float mean = (0);
  float sigma = (50);
  cv::randn(noise, mean, sigma);
  
  cv::imshow("original", image);
  cv::imshow("noise", image + noise);
  
  cv::waitKey(0);
  return 0;
}
