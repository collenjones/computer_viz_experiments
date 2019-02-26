//
//  harris.hpp
//  harris_corner_detector
//
//  Created by Collen Jones on 2/25/19.
//  Copyright © 2019 Collen Jones. All rights reserved.
//

#ifndef harris_hpp
#define harris_hpp

#include <opencv2/core/core.hpp>

namespace harris {
  cv::Mat get_interest_points(const cv::Mat &image, float k, unsigned int descriptor_image_width);
  // cv::Mat suppress_nonmax(const cv::Mat &image, unsigned int min_radius, unsigned int num_to_keep_per_tile);
};

#endif /* harris_hpp */
