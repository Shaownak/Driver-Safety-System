import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import threading
from PIL import Image, ImageTk
import tkinter as tk
import matplotlib.pyplot as plt
import torch.nn as nn
# from transforms import valid_transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms


# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages
    
    

# ====================== Device Initializaion ======================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("CUDA detected! Model is running on GPU.")
else:
    print("CUDA is not available. Model is running on CPU.")
    


# ===================== Desired Comditions =======================
global in_lane, in_depth
in_lane = True
in_depth = True
  

# ===================== Functions ===============================

# ---------- lane Detection ----------   
def detect_lane():
    global in_lane
    
    # setting and directories
    source = r'G:\Driver Safety\Driver_Safety\Thesis\MOV_0086.mp4'
    weights = r'G:\Driver Safety\Driver_Safety\yolopv2.pt'
    imgsz = 640
    

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights)
    

    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    frame_count = 0
    total_fps = 0
    
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)
        t4 = time_synchronized()

        fps = 1 / (t2 - t1) # Forward pass FPS.
        total_fps += fps

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)  # label format
                        # print("Line: ",line)
                        # print("xywh: ", xywh)
                        

                        plot_one_box(xyxy, im0, line_thickness=3)
                        
                        # cv2.line(im0, (xywh[0],xywh[1]), (xywh[2], xywh[3]), (0,255,0), 5)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)
            
            indices=(np.nonzero(ll_seg_mask))
            pixel_coordinates = list(zip(indices[0], indices[1]))
            # pix = (indices[0], indices[1])
            # print(pixel_coordinates)
            
            center_x = im0.shape[1] // 2
            center_y = im0.shape[0] // 2
            
            if pixel_coordinates:  # Make sure the list is not empty
                last_pixel_coordinate = pixel_coordinates[1]
                print("Last pixel coordinate:", last_pixel_coordinate)
                print(f"Center coordinate: {center_x} {center_y}")
                # cv2.line(im0, (center_x, center_y), last_pixel_coordinate, (0,255,255), 5)
                
                img_x = im0.shape[1]
                
                if ((center_x-150)<last_pixel_coordinate[1]) and ((center_x+150)>last_pixel_coordinate[1]):
                    print("In Lane")
                    im0 = cv2.putText(im0, "In Line", (320,100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1,
                                      (0,255,255), 2)
                    in_lane = True
                    print(in_lane)
                else:
                    print("Not In Lane")
                    im0 = cv2.putText(im0, "Not In Line", (320,100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1,
                                      (0,255,255), 2)
                    in_lane = False
                    print(in_lane)
                
            else:
                print("The list is empty, no last pixel coordinate.")
                
            
            cv2.putText(
                im0,
                text=f"YOLOPv2 FPS: {fps:.1f}",
                org=(15, 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.imshow('Image', im0)
            cv2.waitKey(1)

            
    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f"Average FPS: {(total_fps / frame_count):.1f}")
    
    
    
    
    
    
    
# ---------- Depth Estimation ----------

def depth_estimation():
    global in_depth

    # Define the MiDaS model type
    model_type = "MiDaS_small"

    # Load the MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)

    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Open a video capture object
    cap1 = cv2.VideoCapture(r'G:\Driver Safety\Driver_Safety\Thesis\MOV_0086.mp4')

    while True:
        ret, frame = cap1.read()
        
        frame = cv2.resize(frame, (640,480))

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the MiDaS transform
        input_batch = transform(frame_rgb).to(device)

        # Perform depth prediction
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert depth map to grayscale
        depth_map = prediction.cpu().numpy()
        depth_map_gray = (depth_map / depth_map.max() * 255).astype('uint8')

        # Calculate the mean depth value
        mean_depth = np.mean(depth_map)
        
        if mean_depth<300:
            in_depth = False
        else:
            in_depth = True
            
        print(in_depth)

        # Resize depth map to match the video frame size
        depth_map_resized = cv2.resize(depth_map_gray, (frame.shape[1], frame.shape[0]))

        # Create a horizontal stack of the video frame and the depth map
        output_frame = cv2.hconcat([frame, cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)])

        # Display the combined video with mean depth
        cv2.putText(output_frame, f"Mean Depth: {mean_depth:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam + Depth Map', output_frame)

        print("Mean Depth:", mean_depth)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cv2.destroyAllWindows()
    
    
    
    
# ===================== Drowsyness Detection =========================

global drowsy_count, alert_start_time

def drowsyness():
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

        
    
 
# def driver_safety():
#     global in_lane, in_depth
#     # Create a black image (all zeros) with a white text overlay
#     width, height = 640, 480  # Adjust the dimensions as needed
#     image = np.zeros((height, width, 3), dtype=np.uint8)

    
#     if in_lane or in_depth:
#         text = "The driver is unsafe"
#     else:
#         text = "The driver is safe"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0
#     font_thickness = 2
#     font_color = (255, 255, 255)  # White color
#     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#     text_x = int((width - text_size[0]) / 2)
#     text_y = int((height + text_size[1]) / 2)
#     cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

#     # Display the image
#     cv2.imshow('Driver Safety Monitor', image)

#     # Wait for a key press and then close the window
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     print(text)
    
    
    
    
    
    
    


# if __name__ == '__main__':

#     with torch.no_grad():
#             detect_lane()
#             depth_estimation()



# Create a thread for the lane detection function
lane_thread = threading.Thread(target=detect_lane)

# Create a thread for the depth estimation function
depth_thread = threading.Thread(target=depth_estimation)


# Create a thread for the drowsiness detection function
# drowsyness_thread = threading.Thread(target=drowsyness)

# Safety thread
# safety_thread = threading.Thread(target=driver_safety)



if __name__ == '__main__':
    
    # drowsyness_thread.start() # Start the drowsyness detection thread
    # safety_thread.start()
    lane_thread.start()  # Start the lane detection thread
    depth_thread.start()  # Start the depth estimation thread
   
    
    

    # Wait for both threads to finish
    # drowsyness_thread.join()

    # safety_thread.join()
    lane_thread.join()
    depth_thread.join()

    
    # if in_lane==True or in_depth==True:
    #     print("Driver is unsafe")
    #     # driver_safety(message="Driver is unsafe")
    # else:
    #     print("Driver is safe")
    #     # driver_safety(message="Driver is safe")
    
   
    
