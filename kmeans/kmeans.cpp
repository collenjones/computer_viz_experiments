#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

typedef struct t_color_node {
    cv::Mat mean;
    cv::Mat covariance;
    uchar class_id;
    
    t_color_node *left;
    t_color_node *right;
} t_color_node;

std::vector<t_color_node *> get_leaves(t_color_node *root) {
    std::vector<t_color_node *> ret;
    std::queue<t_color_node *> queue;
    queue.push(root);
    
    while (queue.size() > 0) {
        t_color_node *current = queue.front();
        queue.pop();
        
        if (current->left && current->right) {
            queue.push(current->left);
            queue.push(current->right);
            continue;
        }
        ret.push_back(current);
    }
    return ret;
}

cv::Mat get_dominant_palette(std::vector<cv::Vec3b> colors) {
    const int tile_size = 64;
    cv::Mat ret = cv::Mat(tile_size, tile_size * colors.size(), CV_8UC3, cv::Scalar(0));
    
    for (int i = 0; i < colors.size(); i++) {
        cv::Rect rect(i * tile_size, 0, tile_size, tile_size);
        cv::rectangle(ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), CV_FILLED);
    }
    
    return ret;
}

std::vector<cv::Vec3b> get_dominant_colors(t_color_node *root) {
    std::vector<t_color_node *> leaves = get_leaves(root);
    std::vector<cv::Vec3b> ret;
    
    for (int i = 0; i < leaves.size(); i++) {
        cv::Mat mean = leaves[i]->mean;
        ret.push_back(cv::Vec3b(mean.at<double>(0) * 255.0f, mean.at<double>(1) * 255.0f,
                                mean.at<double>(2) * 255.0f));
    }
    return ret;
}

int get_next_class_id(t_color_node *root) {
    int maxid = 0;
    std::queue<t_color_node *> queue;
    queue.push(root);
    
    while (queue.size() > 0) {
        t_color_node *current = queue.front();
        queue.pop();
        
        if (current->class_id > maxid) {
            maxid = current->class_id;
        }
        
        if (current->left != NULL) {
            queue.push(current->left);
        }
        
        if (current->right) {
            queue.push(current->right);
        }
    }
    
    return maxid + 1;
}

cv::Mat get_quantized_image(cv::Mat classes, t_color_node *root) {
    std::vector<t_color_node *> leaves = get_leaves(root);
    
    const int height = classes.rows;
    const int width = classes.cols;
    
    cv::Mat ret(height, width, CV_8UC3, cv::Scalar(0));
    
    for (int y = 0; y < height; y++) {
        uchar *ptrClass = classes.ptr<uchar>(y);
        cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; x++) {
            uchar pixel_class = ptrClass[x];
            for (int i = 0; i < leaves.size(); i++) {
                if (leaves[i]->class_id == pixel_class) {
                    ptr[x] =
                    cv::Vec3b(leaves[i]->mean.at<double>(0) * 255, leaves[i]->mean.at<double>(1) * 255,
                              leaves[i]->mean.at<double>(2) * 255);
                }
            }
        }
    }
    return ret;
}

cv::Mat get_viewable_image(cv::Mat classes) {
    const int height = classes.rows;
    const int width = classes.cols;
    
    const int max_color_count = 12;
    cv::Vec3b *palette = new cv::Vec3b[max_color_count];
    palette[0] = cv::Vec3b(0, 0, 0);
    palette[1] = cv::Vec3b(255, 0, 0);
    palette[2] = cv::Vec3b(0, 255, 0);
    palette[3] = cv::Vec3b(0, 0, 255);
    palette[4] = cv::Vec3b(255, 255, 0);
    palette[5] = cv::Vec3b(0, 255, 255);
    palette[6] = cv::Vec3b(255, 0, 255);
    palette[7] = cv::Vec3b(128, 128, 128);
    palette[8] = cv::Vec3b(128, 255, 128);
    palette[9] = cv::Vec3b(32, 32, 32);
    palette[10] = cv::Vec3b(255, 128, 128);
    palette[11] = cv::Vec3b(128, 128, 255);
    
    cv::Mat ret = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < height; y++) {
        cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
        uchar *ptrClass = classes.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            int color = ptrClass[x];
            if (color >= max_color_count) {
                std::cout << "You should increase the number of predefined colors!" << std::endl;
                continue;
            }
            ptr[x] = palette[color];
        }
    }
    return ret;
}

t_color_node *get_max_eigenvalue_node(t_color_node *current) {
    double max_eigen = -1;
    cv::Mat eigenvalues, eigenvectors;
    std::queue<t_color_node *> queue;
    queue.push(current);
    
    t_color_node *ret = current;
    if (!current->left && !current->right) {
        return current;
    }
    
    while (queue.size() > 0) {
        t_color_node *node = queue.front();
        queue.pop();
        
        if (node->left && node->right) {
            queue.push(node->left);
            queue.push(node->right);
            continue;
        }
        cv::eigen(node->covariance, eigenvalues, eigenvectors);
        double val = eigenvalues.at<double>(0);
        if (val > max_eigen) {
            max_eigen = val;
            ret = node;
        }
    }
    return ret;
}

void partition_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node *node) {
    const int width = img.cols;
    const int height = img.rows;
    const int class_id = node->class_id;
    
    const uchar newidleft = nextid;
    const uchar newidright = nextid + 1;
    
    cv::Mat mean = node->mean;
    cv::Mat covariance = node->covariance;
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covariance, eigenvalues, eigenvectors);
    
    cv::Mat eig = eigenvectors.row(0);
    cv::Mat comparison_value = eig * mean;
    
    node->left = new t_color_node();
    node->right = new t_color_node();
    node->left->class_id = newidleft;
    node->right->class_id = newidright;
    
    for (int y = 0; y < height; y++) {
        cv::Vec3b *ptr = img.ptr<cv::Vec3b>(y);
        uchar *ptrClass = classes.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            if (ptrClass[x] != class_id) {
                continue;
            }
            cv::Vec3b color = ptr[x];
            cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
            scaled.at<double>(0) = color[0] / 255.0f;
            scaled.at<double>(1) = color[1] / 255.0f;
            scaled.at<double>(2) = color[2] / 255.0f;
            
            cv::Mat this_value = eig * scaled;
            if (this_value.at<double>(0, 0) <= comparison_value.at<double>(0, 0)) {
                ptrClass[x] = newidleft;
            } else {
                ptrClass[x] = newidright;
            }
        }
    }
    return;
}

void get_class_mean_covariance(cv::Mat img, cv::Mat classes, t_color_node *node) {
    const int width = img.cols;
    const int height = img.rows;
    const uchar class_id = node->class_id;
    
    cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
    cv::Mat covariance = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));
    
    double pixcount = 0;
    for (int y = 0; y < height; y++) {
        cv::Vec3b *ptr = img.ptr<cv::Vec3b>(y);
        uchar *ptrClass = classes.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            if (ptrClass[x] != class_id) {
                continue;
            }
            cv::Vec3b color = ptr[x];
            cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
            scaled.at<double>(0) = color[0] / 255.0f;
            scaled.at<double>(1) = color[1] / 255.0f;
            scaled.at<double>(2) = color[2] / 255.0f;
            
            mean += scaled;
            covariance += (scaled * scaled.t());
            pixcount++;
        }
    }
    
    covariance -= (mean * mean.t()) / pixcount;
    mean /= pixcount;
    
    node->mean = mean.clone();
    node->covariance = covariance.clone();
    return;
}

std::vector<cv::Vec3b> find_dominant_colors(cv::Mat img, int count) {
    const int width = img.cols;
    const int height = img.rows;
    
    cv::Mat classes = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
    
    t_color_node *root = new t_color_node();
    root->class_id = 1;
    root->left = NULL;
    root->right = NULL;
    
    t_color_node *next = root;
    get_class_mean_covariance(img, classes, root);
    for (int i = 0; i < count; i++) {
        next = get_max_eigenvalue_node(root);
        partition_class(img, classes, get_next_class_id(root), next);
        get_class_mean_covariance(img, classes, next->left);
        get_class_mean_covariance(img, classes, next->right);
    }
    
    std::vector<cv::Vec3b> colors = get_dominant_colors(root);
    
    cv::Mat quantized = get_quantized_image(classes, root);
    cv::Mat viewable = get_viewable_image(classes);
    cv::Mat dom = get_dominant_palette(colors);
    
    cv::imwrite("./classification.png", viewable);
    cv::imwrite("./quantized.png", quantized);
    cv::imwrite("./palette.png", dom);
    
    return colors;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <image> <count>" << std::endl;
        return 0;
    }
    char *filename = argv[1];
    cv::Mat matImage = cv::imread(filename);
    
    if (!matImage.data) {
        std::cout << "Unable to open the file: " << filename << std::endl;
        return -1;
    }
    
    int count = atoi(argv[2]);
    if (count <= 0 || count > 255) {
        std::cout << "The color count needs to be between 1 - 255. You picked " << count << std::endl;
        return -1;
    }
    
    find_dominant_colors(matImage, count);
    return 0;
}

