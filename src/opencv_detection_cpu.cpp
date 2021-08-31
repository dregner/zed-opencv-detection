// Standard includes
#include <stdio.h>
#include <string.h>

// ZED include
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>


cv::Mat slMat2cvMat(sl::Mat &input);

#ifdef HAVE_CUDA

cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat &input);

#endif // HAVE_CUDA

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 256;        // Width of network's input image
int inpHeight = 256;       // Height of network's input image
std::vector<std::string> classes;


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net);

// Sample functions
void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "");

// Sample variables
sl::VIDEO_SETTINGS camera_settings_ = sl::VIDEO_SETTINGS::BRIGHTNESS;
std::string str_camera_settings = "BRIGHTNESS";


sl::Rect selection_rect;
cv::Point origin_rect;

int main(int argc, char **argv) {
    /// Initializing YOLO DETECTION

    // Load names of classes
    std::string classesFile = "/home/nvidia/zed-opencv-detection/yolo_params/cfg/coco_risers.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);
    std::string device = "gpu";

    // Give the configuration and weight files for the model
    cv::String modelConfiguration = "/home/nvidia/zed-opencv-detection/yolo_params/cfg/yolov4_risers.cfg";
    cv::String modelWeights = "/home/nvidia/zed-opencv-detection/yolo_params/weights/yolov4_risers.weights";

    // Load the network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    if (device == "cpu") {
        std::cout << "Using CPU device" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    } else if (device == "gpu") {
        std::cout << "Using GPU device" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    /// INITIALIZING ZED CAM
    // Create a ZED Camera object
    sl::Camera zed;

    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;
    init_parameters.depth_mode = sl::DEPTH_MODE::NONE;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Print camera information
    auto camera_info = zed.getCameraInformation();
    std::cout << std::endl;
    std::cout << "ZED Model                 : " << camera_info.camera_model << std::endl;
    std::cout << "ZED Serial Number         : " << camera_info.serial_number << std::endl;
    std::cout << "ZED Camera Firmware       : " << camera_info.camera_configuration.firmware_version << "/"
              << camera_info.sensors_configuration.firmware_version << std::endl;
    std::cout << "ZED Camera Resolution     : " << camera_info.camera_configuration.resolution.width << "x"
              << camera_info.camera_configuration.resolution.height << std::endl;
    std::cout << "ZED Camera FPS            : " << zed.getInitParameters().camera_fps << std::endl;


#ifndef HAVE_CUDA
    // Create a sl::Mat object (4 channels of type unsigned char) to store the image.
    sl::Mat zed_image(camera_info.camera_configuration.resolution.height,
                      camera_info.camera_configuration.resolution.width, sl::MAT_TYPE::U8_C4);
    // Create an OpenCV Mat that shares sl::Mat data
    cv::Mat image_ocv = slMat2cvMat(zed_image);
#else
    // Create a sl::Mat object (4 channels of type unsigned char) to store the image.
    sl::Mat zed_image(camera_info.camera_configuration.resolution.width,
                      camera_info.camera_configuration.resolution.height, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    // Create an OpenCV Mat that shares sl::Mat data
    cv::Mat image_ocv;
#endif

    cv::Mat blob;


    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state == sl::ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(zed_image, sl::VIEW::LEFT);

            image_ocv = cv::Mat((int) zed_image.getHeight(), (int) zed_image.getWidth(), CV_8UC4,
                                zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::cvtColor(image_ocv, image_ocv, cv::COLOR_RGBA2RGB);
//            gpu_image_ocv.upload(gray);
            // Create a 4D blob from a frame.
            cv::dnn::blobFromImage(image_ocv, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true,
                                   false);

            //Sets the input to the network
            net.setInput(blob);

            // Runs the forward pass to get output of the output layers
            std::vector<cv::Mat> outs;
            net.forward(outs, getOutputsNames(net));

            // Remove the bounding boxes with low confidence
            postprocess(image_ocv, outs);
            // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            std::vector<double> layersTimes;
            double freq = cv::getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            std::string label = cv::format("Inference time for a frame : %.2f ms", t);
            cv::putText(image_ocv, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

//            // Write the frame with the detection boxes
//            cv::Mat detectedFrame;
//            image_ocv.convertTo(detectedFrame, CV_8U);

            //Display the image
            cv::imshow("YOLO DETECTION", image_ocv);
        } else {
            print("Error during capture : ", returned_state);
            break;
        }

        key = cv::waitKey(10);
        // Change camera settings with keyboard
//        updateCameraSettings(key, zed);
    }

    // Exit
    zed.close();
    return EXIT_SUCCESS;
}

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample]";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    else
        std::cout << " ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1:
            cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2:
            cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3:
            cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4:
            cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1:
            cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2:
            cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3:
            cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4:
            cv_type = CV_8UC4;
            break;
        default:
            break;
    }
    return cv_type;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame) {
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)),
              cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net) {
    static std::vector<cv::String> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat &input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()),
                   input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}

#ifdef HAVE_CUDA

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat &input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()),
                            input.getPtr<sl::uchar1>(sl::MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}

#endif
