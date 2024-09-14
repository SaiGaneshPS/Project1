import cv2
import numpy as np
from text_detection import decode_predictions

def load_video(capture, net):
    ret, frame = capture.read()
    if not ret:
        return None, None, None

    image_frame = frame.copy()  # Copy for later use without bounding boxes
    
    (H, W) = frame.shape[:2]
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    resized_frame = cv2.resize(frame, (newW, newH))

    # Prepare the frame for the EAST model
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(output_layers)

    # Decode the detections and extract bounding boxes
    boxes = decode_predictions(scores, geometry, min_confidence=0.5)

    # Merge boxes to create one large bounding box
    if boxes:
        boxes_array = np.array(boxes)
        x_min = np.min(boxes_array[:, 0])
        y_min = np.min(boxes_array[:, 1])
        x_max = np.max(boxes_array[:, 2])
        y_max = np.max(boxes_array[:, 3])

        startX = int(x_min * rW)
        startY = int(y_min * rH)
        endX = int(x_max * rW)
        endY = int(y_max * rH)

        return frame, (startX, startY, endX, endY), image_frame
    return frame, None, image_frame
