import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect_lanes(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest
    height, width = edges.shape
    vertices = np.array([[(100, height), (width // 2 - 50, height // 2 + 50), (width // 2 + 50, height // 2 + 50), (width - 50, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    # Create a blank image to draw lines on
    line_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the detected lines on the blank image
    draw_lines(line_image, lines)

    # Combine the lane lines image with the original image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

# Test the lane detection on a video stream or an image
cap = cv2.VideoCapture(r'G:\Driver Safety\Driver_Safety\test_video\test.mp4')  # Use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_lanes = detect_lanes(frame)
    cv2.imshow('Lane Detection', frame_with_lanes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
