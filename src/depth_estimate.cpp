
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
    init_parameters.sdk_verbose = false;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = sl::UNIT::METER; // Use meter units (for depth measurements)


    // Open the camera
    if (zed.open(init_parameters) != sl::ERROR_CODE::SUCCESS) {
        return EXIT_FAILURE;
    }

    // Create a sl::Mat object (4 channels of type unsigned char) to store the image.
    auto camera_info = zed.getCameraInformation();
    sl::Mat zed_image(camera_info.camera_configuration.resolution.width,
                      camera_info.camera_configuration.resolution.height, sl::MAT_TYPE::F32_C1, sl::MEM::CPU);
    // Create an OpenCV Mat that shares sl::Mat data
    cv::Mat image_ocv  =  slMat2cvMat(zed_image);
    sl::Mat depth;
    float depth_value;
	

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {

            // Retrieve left image
            zed.retrieveImage(zed_image, sl::VIEW::DEPTH, sl::MEM::CPU);

            //image_ocv_gpu = cv::cuda::GpuMat((int) zed_image.getHeight(), (int) zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::GPU));
            image_ocv = slMat2cvMat(zed_image);

            // Retrieve depth measure
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
            depth.getValue(110, 130, &depth_value);
            std::cout << "Depth " << depth_value << std::endl;




            //Display the image
            cv::imshow("Depth estimate", image_ocv);
        } else {
            break;
        }

        key = cv::waitKey(5);
    }

    // Exit
    zed.close();
    return EXIT_SUCCESS;
}




