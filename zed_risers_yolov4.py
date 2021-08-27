import os
import time
import numpy as np
import glob
import sys
import os.path
import imutils


import cv2  # conda install -c conda-forge opencv=4.3.0
import pyzed.sl as sl

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

net_yolov4 = cv2.dnn.readNet('weights/yolov4_risers_final.weights', 'cfg/yolov4_risers.cfg')
net_yolov4.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_yolov4.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


class_names = []
coconames='cfg/coco.names'
with open(coconames,'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
    

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.5
model = cv2.dnn_DetectionModel(net_yolov4)
model.setInputParams(size=(416, 416), scale=1/255)
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

tiempo_yolov4 = []
FPS_total_yolov4=[] 

start_drawing = time.time()

def main():
    print("Running...")
    init = sl.InitParameters()
    zed = sl.Camera()
    if not zed.is_opened():
        print("Opening ZED Camera...")
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    # Create an RGBA sl.Mat object
    image_zed = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    # Retrieve data in a numpy array with get_data()
    image_ocv = image_zed.get_data()

    key = ''
    while key != 113:  # for 'q' key
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()
            classes, scores, boxes = model.detect(image_ocv, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB).astype(np.float32)
            # image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_RGB2RGBA).astype(np.float32)

            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(image_ocv, box, color, 6)
                cv2.putText(image_ocv, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                cv2.rectangle(image_ocv, box, color, 6)
                cv2.putText(image_ocv, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)

            img_resize = imutils.resize(image_ocv, width=800)
            cv2.imshow("ZED Detection", img_resize)
            key = cv2.waitKey(5)         
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()


