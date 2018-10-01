#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>

/**
 * 1. Pad with zeroes
 * 2. Support grayscale and color images * 3. Support arbitrary shaped
 * odd-dimension filters (e.g. 7x9 but not 4x5)
 * 4. Return an error message for even filters as their output is undefined
 * 5. Return an identical image with an identity filter
 * 6. Return a filtered image which is the same resolution as the original image
 */
cv::Mat filter(cv::Mat &original_image, cv::Mat &kernel) {
    if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0) {
        std::cout << "kernel rows and columns cannot be even" << std::endl;
        exit(-1);
    }
    std::vector<int> kernel_midpoints = {kernel.rows / 2, kernel.cols / 2};
    cv::Mat filtered_image =
    cv::Mat::zeros(original_image.size(), original_image.type());
    for (int row = 0; row < original_image.rows; row++) {
        for (int col = 0; col < original_image.cols; col++) {
            cv::Vec3b color = original_image.at<cv::Vec3b>(col, row);
            // Pad edges with zeroes
            if (row < kernel_midpoints[0] ||
                row > original_image.rows - 1 - kernel_midpoints[0] ||
                col < kernel_midpoints[1] ||
                col > original_image.cols - 1 - kernel_midpoints[1]) {
                color = {0, 0, 0};
            } else {
                cv::Vec3b total = {0, 0, 0};
                for (int kernel_row = 0; kernel_row < kernel.rows; kernel_row++) {
                    for (int kernel_col = 0; kernel_col < kernel.cols; kernel_col++) {
                        // Use a pixel offset of -1 to 1 for a 3x3 matrix for both rows and
                        // columns
                        int image_row_index = kernel_row - kernel.rows / 2;
                        int image_col_index = kernel_col - kernel.cols / 2;
                        float kernel_value = kernel.at<float>(kernel_row, kernel_col);
                        cv::Vec3b original_pixel = original_image.at<cv::Vec3b>(
                                                                                cv::Point(col + image_col_index, row + image_row_index));
                        total += original_pixel * kernel_value;
                    }
                }
                color = total;
            }
            filtered_image.at<cv::Vec3b>(cv::Point(col, row)) = color;
        }
    }
    return filtered_image;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: image_filter <path to image>" << std::endl;
        return -1;
    }
    
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!image.data) {
        std::cout << "Could not open file or find the image: " << argv[1]
        << std::endl;
        return -1;
    }
    
    cv::Mat IDENTITY_KERNEL = cv::Mat::zeros(3, 3, CV_32F);
    IDENTITY_KERNEL.at<float>(1, 1) = 1.f;
    
    cv::Mat BOX_FILTER(3, 3, CV_32F);
    BOX_FILTER.setTo(cv::Scalar(1.f / 9.f));
    
    cv::Mat LARGE_BOX_FILTER(7, 9, CV_32F);
    LARGE_BOX_FILTER.setTo(cv::Scalar(1.f / (7. * 9.)));
    
    cv::Mat LEFT_SOBEL =
    (cv::Mat_<float>(3, 3) << 1., 0., -1., 2., 0., -2., 1., 0., -1.);
    
    cv::imshow("Original image", image);
    cv::imshow("Identity Filter", filter(image, IDENTITY_KERNEL));
    cv::imshow("Box Filter", filter(image, BOX_FILTER));
    cv::imshow("Large Box Filter", filter(image, LARGE_BOX_FILTER));
    cv::imshow("Left Sobel", filter(image, LEFT_SOBEL));
    
    cv::waitKey(0);
    return 0;
}

