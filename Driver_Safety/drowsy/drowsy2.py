import cv2
import torch
from PIL import Image, ImageTk
import tkinter as tk
import matplotlib.pyplot as plt
import torch.nn as nn
from transforms import valid_transforms
import time


softmax = nn.Softmax(dim=1)

label_dict = {'awake': 1, 'drowsy': 0}
inv_label_dict = {i: j for j, i in label_dict.items()}

cam_port = 0

scripted_module = torch.jit.load("model.pt")
cap = cv2.VideoCapture('drowsy.mp4')

# Create a tkinter window
window = tk.Tk()
window.title("Webcam")

# Create a label to display the frame
label = tk.Label(window)
label.pack()

# Variables for alert system
drowsy_count = 0  # Counter for consecutive drowsy outputs
alert_start_time = 0  # Time when the alert started

# Create a label for the alert
alert_label = tk.Label(window, fg="red", font=("Arial", 18))
alert_label.pack()

    
while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Perform any necessary pre-processing on the frame (e.g., resizing, normalization)
    # Pre-process the frame to match the input format expected by your PyTorch model
    
    # if not isinstance(frame, np.ndarray):
    #     frame = np.array(frame)


    # valid_transforms = A.Compose([
    #     A.Resize(256, 256),
    #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ])
    
    if frame is not None:
        transformed_img = valid_transforms(image=frame)

        img = transformed_img['image'].unsqueeze(0) / 255.0

        out = scripted_module(img.cuda())

        soft_out = softmax(out)

        label_text = inv_label_dict[soft_out.argmax().item()]

        if label_text == 'drowsy':
            if drowsy_count == 0:
                alert_start_time = time.time()  # Start the alert timer
            drowsy_count += 1
        else:
            drowsy_count = 0  # Reset the counter if not drowsy

        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)  # Display the frame using cv2.imshow()

        if drowsy_count >= 3:  # Check if drowsy outputs are continuous for 5 or more frames
            alert_duration = time.time() - alert_start_time  # Calculate the duration of the alert
            if alert_duration >= 3.0:  # Check if the alert has been continuous for 5 seconds
                print("Drowsiness Alert!")  # Print an alert message
        else:
            print("")  # Print an empty line if not enough consecutive drowsy outputs
            
    else:
        continue

    

    if cv2.waitKey(10) & 0xFF == ord('q'):  # Check for the 'q' key to exit the loop
        break

cv2.destroyAllWindows()  # Close all OpenCV windows when done
cap.release()  # Release the video capture object
