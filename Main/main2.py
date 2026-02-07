import cv2
import torch
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time
from functools import partial
import matplotlib.pyplot as plt
import torch.nn as nn
from transforms import valid_transforms
import time


# from utils.utils import \
#     time_synchronized,select_device, increment_path,\
#     scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
#     driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
#     AverageMeter,\
#     LoadImages


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("CUDA detected! Model is running on GPU.")
else:
    print("CUDA is not available. Model is running on CPU.")

# Define global variables for sharing data between functions
lane_detection_result = None
drowsiness_detection_result = None
depth_estimation_result = None

# Function for lane detection
# def lane_detection():
    

    # def make_parser():
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--weights', nargs='+', type=str, default=r'G:\Driver Safety\Driver_Safety\yolopv2.pt', help='model.pt path(s)')
    #     parser.add_argument('--source', type=str, default=r'G:\Driver Safety\Driver_Safety\test_video\test.mp4', help='source')  # file/folder, 0 for webcam
    #     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #     parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    #     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    #     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    #     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    #     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    #     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    #     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    #     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    #     parser.add_argument('--name', default='exp', help='save results to project/name')
    #     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #     return parser


    # def detect():
    #     # setting and directories
    #     source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    #     save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    #     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #     inf_time = AverageMeter()
    #     waste_time = AverageMeter()
    #     nms_time = AverageMeter()

    #     # Load model
    #     stride =32
    #     model  = torch.jit.load(weights)
    #     device = select_device(opt.device)
    #     half = device.type != 'cpu'  # half precision only supported on CUDA
    #     model = model.to(device)

    #     if half:
    #         model.half()  # to FP16  
    #     model.eval()

    #     # Set Dataloader
    #     vid_path, vid_writer = None, None
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    #     # Run inference
    #     if device.type != 'cpu':
    #         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #     t0 = time.time()
    #     frame_count = 0
    #     total_fps = 0
    #     for path, img, im0s, vid_cap in dataset:
    #         frame_count += 1
    #         img = torch.from_numpy(img).to(device)
    #         img = img.half() if half else img.float()  # uint8 to fp16/32
    #         img /= 255.0  # 0 - 255 to 0.0 - 1.0

    #         if img.ndimension() == 3:
    #             img = img.unsqueeze(0)

    #         # Inference
    #         t1 = time_synchronized()
    #         [pred,anchor_grid],seg,ll= model(img)
    #         t2 = time_synchronized()

    #         # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
    #         # but this problem will not appear in offical version 
    #         tw1 = time_synchronized()
    #         pred = split_for_trace_model(pred,anchor_grid)
    #         tw2 = time_synchronized()

    #         # Apply NMS
    #         t3 = time_synchronized()
    #         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #         t4 = time_synchronized()

    #         fps = 1 / (t2 - t1) # Forward pass FPS.
    #         total_fps += fps

    #         da_seg_mask = driving_area_mask(seg)
    #         ll_seg_mask = lane_line_mask(ll)

    #         # Process detections
    #         for i, det in enumerate(pred):  # detections per image
            
    #             p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

    #             p = Path(p)  # to Path
    #             save_path = str(save_dir / p.name)  # img.jpg
    #             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
    #             s += '%gx%g ' % img.shape[2:]  # print string
    #             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #             if len(det):
    #                 # Rescale boxes from img_size to im0 size
    #                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    #                 # Print results
    #                 for c in det[:, -1].unique():
    #                     n = (det[:, -1] == c).sum()  # detections per class
    #                     #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    #                 # Write results
    #                 for *xyxy, conf, cls in reversed(det):
    #                     if save_txt:  # Write to file
    #                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
    #                         with open(txt_path + '.txt', 'a') as f:
    #                             f.write(('%g ' * len(line)).rstrip() % line + '\n')

    #                     if save_img :  # Add bbox to image
    #                         plot_one_box(xyxy, im0, line_thickness=3)

    #             # Print time (inference)
    #             print(f'{s}Done. ({t2 - t1:.3f}s)')
    #             show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)
    #             cv2.putText(
    #                 im0,
    #                 text=f"YOLOPv2 FPS: {fps:.1f}",
    #                 org=(15, 35),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=1,
    #                 color=(0, 0, 255),
    #                 thickness=2,
    #                 lineType=cv2.LINE_AA
    #             )
    #             cv2.imshow('Lane Detection', im0)
    #             cv2.waitKey(1)

    #             # Save results (image with detections)
    #             if save_img:
    #                 if dataset.mode == 'image':
    #                     cv2.imwrite(save_path, im0)
    #                     print(f" The image with the result is saved in: {save_path}")
    #                 else:  # 'video' or 'stream'
    #                     if vid_path != save_path:  # new video
    #                         vid_path = save_path
    #                         if isinstance(vid_writer, cv2.VideoWriter):
    #                             vid_writer.release()  # release previous video writer
    #                         if vid_cap:  # video
    #                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                             w, h = im0.shape[1], im0.shape[0]
    #                         else:  # stream
    #                             fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                             save_path += '.mp4'
    #                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                     vid_writer.write(im0)

    #     inf_time.update(t2-t1,img.size(0))
    #     nms_time.update(t4-t3,img.size(0))
    #     waste_time.update(tw2-tw1,img.size(0))
    #     print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    #     print(f'Done. ({time.time() - t0:.3f}s)')
    #     print(f"Average FPS: {(total_fps / frame_count):.1f}")

# Function for drowsiness detection
def drowsiness_detection():
    softmax = nn.Softmax(dim=1)

    label_dict = {'awake': 1, 'drowsy': 0}
    inv_label_dict = {i: j for j, i in label_dict.items()}

    cam_port = 0

    scripted_module = torch.jit.load(r"G:\Driver Safety\Driver_Safety\model.pt")
    cap = cv2.VideoCapture(r'G:\Driver Safety\Driver_Safety\drowsy.mp4')

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

    def show_frame():
        global drowsy_count, alert_start_time

        ret, frame = cap.read()  # Read a frame from the webcam

        # Perform any necessary pre-processing on the frame (e.g., resizing, normalization)
        # Pre-process the frame to match the input format expected by your PyTorch model
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

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_tk = ImageTk.PhotoImage(frame_pil)

        label.config(image=frame_tk)
        label.image = frame_tk

        if drowsy_count >= 3:  # Check if drowsy outputs are continuous for 5 or more frames
            alert_duration = time.time() - alert_start_time  # Calculate the duration of the alert
            if alert_duration >= 3.0:  # Check if the alert has been continuous for 5 seconds
                alert_label.config(text="Drowsiness Alert!")  # Update the alert label
            else:
                alert_label.config(text="")  # Clear the alert label if the duration is less than 5 seconds
        else:
            alert_label.config(text="")  # Clear the alert label if not enough consecutive drowsy outputs

        label.after(10, show_frame)

    show_frame()

    window.mainloop()

    cap.release()
    cv2.destroyAllWindows()

# Function for depth estimation
def depth_estimation():
    model_type = "MiDaS_small" 
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    cap = cv2.VideoCapture(r'G:\Driver Safety\Driver_Safety\test_video\test.mp4')
    while True:
        ret, frame = cap.read()
        print(frame.shape)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(frame_rgb).to(device)

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

        # Resize depth map to match webcam video size
        depth_map_resized = cv2.resize(depth_map_gray, (frame.shape[1], frame.shape[0]))

        # Create a horizontal stack of webcam video and depth map
        output_frame = cv2.hconcat([frame, cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)])

        # Display the combined video
        cv2.imshow('Webcam + Depth Map', output_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start all three processes
def start_driver_safety_system():
    global lane_detection_result, drowsiness_detection_result, depth_estimation_result

    # Create separate threads for each function
    # lane_thread = threading.Thread(target=lane_detection)
    drowsy_thread = threading.Thread(target=drowsiness_detection)
    depth_thread = threading.Thread(target=depth_estimation)

    # Start the threads
    # lane_thread.start()
    drowsy_thread.start()
    depth_thread.start()

    # Wait for the threads to finish
    # lane_thread.join()
    drowsy_thread.join()
    depth_thread.join()

# Main function to run the driver safety system
def main():
    # Create a tkinter window for displaying results
    main_window = tk.Tk()
    main_window.title("Driver Safety System")

    # Create labels for displaying results
    # lane_label = tk.Label(main_window, text="Lane Detection Result")
    # lane_label.pack()

    drowsy_label = tk.Label(main_window, text="Drowsiness Detection Result")
    drowsy_label.pack()

    depth_label = tk.Label(main_window, text="Depth Estimation Result")
    depth_label.pack()

    # Create a button to start the driver safety system
    start_button = tk.Button(main_window, text="Start Driver Safety System", command=start_driver_safety_system)
    start_button.pack()

    # Start the tkinter main loop
    main_window.mainloop()

if __name__ == '__main__':
    main()
