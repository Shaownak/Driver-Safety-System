import torch
import cv2
import numpy as np

# Load the TorchScript model (replace 'your_model.pt' with the path to your model)
model = torch.jit.load('model.pt')
model.eval()  # Set the model to evaluation mode

# Specify the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the specified device

# Open a video capture object (replace 'your_video.mp4' with the path to your video)
cap = cv2.VideoCapture('drowsy.mp4')

while True:
    ret, frame = cap.read()  # Read a frame from the video

    if not ret:
        break  # Break the loop if the video has ended

    # Preprocess the frame (resize, normalize, etc.) to match the model's input requirements
    # Replace this with your preprocessing code as needed
    # Make sure 'frame' is a NumPy array in the correct format for your model

    # Convert the frame to a PyTorch tensor and move it to the specified device
    input_tensor = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    input_tensor = input_tensor.to(device)

    # Perform object detection inference
    with torch.no_grad():
        detections = model(input_tensor)

    # Process the detection results (replace this with your post-processing code)
    # 'detections' should contain the predicted bounding boxes, labels, and scores

    # Draw bounding boxes on the frame based on the detection results
    for detection in detections:
        # x1, y1, x2, y2, confidence, class_id = detection.tolist()

        # # Draw a bounding box
        # color = (0, 255, 0)  # Green color for the bounding box
        # thickness = 2  # Line thickness
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # # Add class label and confidence score
        # label = f'Class {int(class_id)}: {confidence:.2f}'
        # cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        print(detection)


    # Display the frame with detections
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
