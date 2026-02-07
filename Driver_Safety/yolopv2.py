import cv2
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("CUDA detected! Model is running on GPU.")
else:
    print("CUDA is not available. Model is running on CPU.")
    
    
# model_type = "YOLOPv2"
model_path = r"G:\Driver Safety\Driver_Safety\yolopv2.pt" 
# model = torch.hub.load("CAIC-AD/YOLOPv2", model_type)
model = torch.jit.load(model_path)
model.to(device)


cap = cv2.VideoCapture(r'G:\Driver Safety\Driver_Safety\test_video\test.mp4')


while True:
    ret, frame = cap.read()  
    print(frame.shape)


    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    
        
    

cap.release()
cv2.destroyAllWindows()