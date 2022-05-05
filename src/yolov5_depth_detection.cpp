#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.hpp"

#include <sl/Camera.hpp>


#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) +
                               1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output,
                 int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);
}

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

std::vector<sl::uint2> cvt(const cv::Rect &bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

int main(int argc, char **argv) {
    static double previous_time = sl::Timestamp().data_ns;
    std::string wts_name = "";
    std::string engine_name = "/home/jetson/Documents/zed-opencv-detection/yolo_params/weights/yolov5s6.engine";
    bool is_p6 = true;

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 60;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.depth_minimum_distance = 0.7;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    /// Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720),
                                 std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    assert(BATCH_SIZE == 1); // This sample only support batch 1 for now

    sl::Mat left_sl, depth;
    cv::Mat left_cv_rgb;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    float depth_value;
//    while (viewer.isAvailable()) {
    while (zed.isOpened()) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            double dt = previous_time - sl::Timestamp().data_ns;

            zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH, sl::MEM::CPU);
            // Preparing inference
            cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
            cv::cvtColor(left_cv_rgba, left_cv_rgb, cv::COLOR_BGRA2BGR);
//            if (left_cv_rgb.empty()) continue;
            cv::Mat pr_img = preprocess_img(left_cv_rgb, INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = 0;
            int batch = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar *uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[batch * 3 * INPUT_H * INPUT_W + i] = (float) uc_pixel[2] / 255.0;
                    data[batch * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float) uc_pixel[1] / 255.0;
                    data[batch * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float) uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }

            // Running inference
            doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
            std::vector<std::vector<Yolo::Detection >> batch_res(BATCH_SIZE);
            auto &res = batch_res[batch];
            nms(res, &prob[batch * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it: res) {
                sl::CustomBoxObjectData tmp;
                cv::Rect r = get_rect(left_cv_rgb, it.bbox);
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.conf;
                tmp.label = (int) it.class_id;
                tmp.bounding_box_2d = cvt(r);
                tmp.is_grounded = ((int) it.class_id ==
                                   0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space
                objects_in.push_back(tmp);
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);


            // Displaying 'raw' objects
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(left_cv_rgb, res[j].bbox);
                cv::rectangle(left_cv_rgb, r, cv::Scalar(0x27, 0xC1, 0x36), 2);

                //if (res[j].class_id == 0) {
                    depth.getValue(r.x / 2, r.y / 2, &depth_value, sl::MEM::GPU);
                    cv::putText(left_cv_rgb,
                                std::to_string((int) res[j].class_id) + " x " + std::to_string((float) depth_value) +
                                " m", cv::Point(r.x, r.y - 2),
                                cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                    std::cout << "Pixel pos - " << res[j].class_id << ": " << r.width/2 << " x " << r.height/2 << std::endl;
                //} else {
                    //cv::putText(left_cv_rgb,
                               // std::to_string((int) res[j].class_id) + " x " + std::to_string((float) res[j].conf),
                                //cv::Point(r.x, r.y - 2),
                                //cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
             //   }
            }
            previous_time = sl::Timestamp().data_ns;
            std::string label = cv::format("Inference time for a frame : %.2f ms", dt);
            cv::putText(left_cv_rgb, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

            cv::imshow("Yolo V5 Detection", left_cv_rgb);
            cv::waitKey(5);

        }
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
