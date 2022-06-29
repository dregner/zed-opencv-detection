
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
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_units = sl::UNIT::METER; // Use meter units (for depth measurements)
    init_parameters.depth_minimum_distance = 0.4;
    init_parameters.depth_maximum_distance = 20;
    

    // Open the camera
    if (zed.open(init_parameters) != sl::ERROR_CODE::SUCCESS) {
        return EXIT_FAILURE;
    }
    sl::RuntimeParameters runParameters;
    runParameters.sensing_mode = sl::SENSING_MODE::FILL;
//    runParameters.confidence_threshold = 50;
//    runParameters.texture_confidence_threshold = 50;
    // Create a sl::Mat object (4 channels of type unsigned char) to store the image.
    auto camera_info = zed.getCameraInformation();
    sl::Mat zed_image(camera_info.camera_configuration.resolution.width,
                      camera_info.camera_configuration.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);
    // Create an OpenCV Mat that shares sl::Mat data
    cv::cuda::GpuMat img_ocv_gpu = slMat2cvMatGPU(zed_image);
    cv::Mat img_ocv, depth_ocv;
    sl::Mat depth, depth_view;
    float depth_value[10], accum;
    int x = 640, y = 360;
    std::string input;

    //! Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        if (zed.grab(runParameters) == sl::ERROR_CODE::SUCCESS) {
            if(key == 'c'){
                std::cout << "Add X value from 0 to " << camera_info.camera_configuration.resolution.width << std::endl;
                std::cin >> x;
                std::cout << "Add Y value from 0 to " << camera_info.camera_configuration.resolution.height << std::endl;
                std::cin >> y;
            }
            //! Retrieve left image
            zed.retrieveImage(zed_image, sl::VIEW::LEFT, sl::MEM::GPU);
            zed.retrieveImage(depth_view, sl::VIEW::DEPTH, sl::MEM::CPU);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
            accum = 0;
            for (size_t k = 0; k < sizeof(depth_value) / sizeof(depth_value[0]); k++) {
                depth.getValue(x, y, &depth_value[k], sl::MEM::CPU);
                if(isValidMeasure(depth_value[k])) {
                    accum += depth_value[k] * depth_value[k];
                }
            }
            double distance = sqrt(accum/sizeof(depth_value) / sizeof(depth_value[0]));

//            img_ocv_gpu = slMat2cvMatGPU(zed_image);
//            img_ocv_gpu.download(img_ocv);
            img_ocv = slMat2cvMat(depth_view);
            cv::circle(img_ocv, cv::Point(x, y), 10, cv::Scalar(255, 0, 255), cv::LINE_4);
            cv::putText(img_ocv, std::to_string(distance) + " m", cv::Point(50,  100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            // Retrieve depth measure
//            depth.getValue(camera_info.camera_configuration.resolution.width/2, camera_info.camera_configuration.resolution.height/2, &depth_value);
//            std::cout << "Depth midle: " << depth_value << std::endl;
            //Display the image
            cv::imshow("Depth estimate", img_ocv);
        } else {
            break;
        }

        key = cv::waitKey(5);
    }

    // Exit1
    zed.close();
    return EXIT_SUCCESS;
}




