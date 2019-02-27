//
//  harris.hpp
//  harris_corner_detector
//
//  Created by Collen Jones on 2/25/19.
//  Copyright © 2019 Collen Jones. All rights reserved.
//

#ifndef harris_hpp
#define harris_hpp

#include <vector>
#include <opencv2/core/core.hpp>

namespace harris {
  struct InterestPoint {
    float corner_value;
    cv::Point point;
  };
  
  cv::Mat get_interest_points(const cv::Mat &image, float k);
  std::vector<harris::InterestPoint> suppress_nonmax(const cv::Mat &interest_points, unsigned int num_per_tile, unsigned int min_pixel_radius);
};

#endif /* harris_hpp */
