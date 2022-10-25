
// ZED include
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utils.hpp>
#include "chrono"
typedef std::chrono::time_point<std::chrono::high_resolution_clock> timer;
typedef std::chrono::duration<float> duration;

int main(int argc, char **argv) {
    /// INITIALIZING ZED CAM
    // Create a ZED Camera object
    sl::Camera zed;

    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
    init_parameters.coordinate_units = sl::UNIT::METER; // Use meter units (for depth measurements)
    init_parameters.depth_minimum_distance = 0.4;
    init_parameters.depth_maximum_distance = 20;


    // Open the camera
    if (zed.open(init_parameters) != sl::ERROR_CODE::SUCCESS) {
        return EXIT_FAILURE;
    }
    sl::RuntimeParameters runParameters;
    runParameters.sensing_mode = sl::SENSING_MODE::FILL;


// Enable recording with the filename specified in argument
    sl::String path_output = "/home/jetson/Documents/zed_svo/record_neural.svo";
    auto returned_state = zed.enableRecording(
            sl::RecordingParameters(path_output, sl::SVO_COMPRESSION_MODE::H264_LOSSLESS));
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Recording ZED : " << returned_state << std::endl;
        zed.

                close();

        return EXIT_FAILURE;
    }

// Start recording SVO, stop with Ctrl-C command
    std::cout << "SVO is Recording, use Ctrl-C to stop." << std::endl;
    SetCtrlHandler();

    int frames_recorded = 0;
    sl::RecordingStatus rec_status;

    while (!exit_app) {
        timer frame_start = std::chrono::high_resolution_clock::now();
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            rec_status = zed.getRecordingStatus();
            if (rec_status.status){
                frames_recorded++;
             }
            timer frame_end = std::chrono::high_resolution_clock::now();
            duration frame_duration = frame_end - frame_start;
            std::cout << "Frame count: " << std::to_string(frames_recorded) << std::endl;
            std::cout << "Frame time: " << frame_duration.count()*1000 << " ms" <<std::endl;
        }
    }

// Stop recording
    zed.disableRecording();

    zed.close();

    return EXIT_SUCCESS;
}




