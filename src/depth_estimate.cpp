
// ZED include
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utils.hpp>


// Sample variables
sl::VIDEO_SETTINGS camera_settings_ = sl::VIDEO_SETTINGS::BRIGHTNESS;
std::string str_camera_settings = "BRIGHTNESS";


sl::Rect selection_rect;
cv::Point origin_rect;

int main(int argc, char **argv) {
    /// INITIALIZING ZED CAM
    // Create a ZED Camera object
    sl::Camera zed;

    sl::InitParameters init_parameters;
<<<<<<< HEAD
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
=======
    init_parameters.sdk_verbose = false;
    init_parameters.camera_resolution = sl::RESOLUTION::VGA;
>>>>>>> cb0ebcacfd2dbacadb39d0c32283e0ee0f534656
    init_parameters.camera_fps = 60;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = sl::UNIT::METER; // Use meter units (for depth measurements)
    init_parameters.depth_minimum_distance = 0.9;
    init_parameters.depth_maximum_distance = 20;
    

    // Open the camera
    if (zed.open(init_parameters) != sl::ERROR_CODE::SUCCESS) {
        return EXIT_FAILURE;
    }
    sl::RuntimeParameters runParameters;
    runParameters.sensing_mode = sl::SENSING_MODE::FILL;
    runParameters.confidence_threshold = 50;
    runParameters.texture_confidence_threshold = 100;
    // Create a sl::Mat object (4 channels of type unsigned char) to store the image.
    auto camera_info = zed.getCameraInformation();
    sl::Mat zed_image(camera_info.camera_configuration.resolution.width,
                      camera_info.camera_configuration.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);
    // Create an OpenCV Mat that shares sl::Mat data
    cv::cuda::GpuMat img_ocv_gpu = slMat2cvMatGPU(zed_image);
    cv::Mat img_ocv;
    sl::Mat depth;
    float depth_value;
    int x = 110, y = 130;
    std::string input;

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            std::cin >> input;
            if(key == 'c'){
                std::cout << "Add X value from 0 to " << camera_info.camera_configuration.resolution.width << std::endl;
                std::cin >> x;
                std::cout << "Add Y value from 0 to " << camera_info.camera_configuration.resolution.height << std::endl;
                std::cin >> y;
            }
            // Retrieve left image
            zed.retrieveImage(zed_image, sl::VIEW::LEFT, sl::MEM::CPU);

            //image_ocv_gpu = cv::cuda::GpuMat((int) zed_image.getHeight(), (int) zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::GPU));
            img_ocv_gpu = slMat2cvMatGPU(zed_image);
            cv::circle(img_ocv_gpu, cv::Point(x, y), 100, cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(img_ocv_gpu, std::to_string(depth_value) + " m", cv::Point(x + 100, y + 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            // Retrieve depth measure
            img_ocv_gpu.download(img_ocv);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
<<<<<<< HEAD
            depth.getValue(camera_info.camera_configuration.resolution.width/2, camera_info.camera_configuration.resolution.height/2, &depth_value);
            std::cout << "Depth midle: " << depth_value << std::endl;
=======
            depth.getValue(110, 130, &depth_value);
            std::cout << "Depth (" <<x << ", " << y << ") - " << depth_value <<" m" << std::endl;
>>>>>>> cb0ebcacfd2dbacadb39d0c32283e0ee0f534656




            //Display the image
            cv::imshow("Depth estimate", img_ocv);
        } else {
            break;
        }

        key = cv::waitKey(5);
    }

    // Exit
    zed.close();
    return EXIT_SUCCESS;
}




