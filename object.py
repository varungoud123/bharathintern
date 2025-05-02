import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained object detection model (SSD MobileNet v2)
model_path = r"C:\Users\Medipelly Varun\Desktop\object detection\saved_model"  # Use raw string or double backslashes
model = tf.saved_model.load(model_path)
detect_fn = model.signatures['serving_default']

# Load the full COCO dataset label map
category_index = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                  6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                  11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
                  15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
                  20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
                  25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                  32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
                  36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                  40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
                  43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
                  48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
                  53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
                  57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
                  62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                  67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                  75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
                  79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
                  85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
                  89: 'hair drier', 90: 'toothbrush'}

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' for the primary webcam

# Setup a figure for matplotlib display
plt.ion()  # Interactive mode for live updates

# Loop to process each frame from the webcam
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the frame to improve performance (e.g., 640x480)
    resized_frame = cv2.resize(frame, (640, 480))

    # Convert the resized frame to a tensor for model input
    input_tensor = tf.convert_to_tensor(np.expand_dims(resized_frame, 0), dtype=tf.uint8)

    # Perform object detection
    detections = detect_fn(input_tensor)

    # Process detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detection_classes = detections['detection_classes'].astype(np.int32)
    detection_boxes = detections['detection_boxes']
    detection_scores = detections['detection_scores']

    # Loop over the detections and draw bounding boxes
    for i in range(num_detections):
        if detection_scores[i] > 0.3:  # Confidence threshold
            class_name = category_index.get(detection_classes[i], 'N/A')
            box = detection_boxes[i]

            # Convert box coordinates from relative values to pixel coordinates
            (h, w) = resized_frame.shape[:2]
            (startY, startX, endY, endX) = (int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w))

            # Draw bounding box and label
            cv2.rectangle(resized_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{class_name}: {detection_scores[i]:.2f}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB (Matplotlib expects RGB images)
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Matplotlib
    plt.imshow(frame_rgb)
    plt.title("Real-Time Object Detection")
    plt.draw()
    plt.pause(0.001)  # Pause for a short period to allow the figure to update

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
plt.close()

