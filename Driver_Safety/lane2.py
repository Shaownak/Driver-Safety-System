import cv2
import numpy as np

class LaneTracker:
    def __init__(self):
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman_filter.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
        self.kalman_filter.measurementNoiseCov = 1e-3 * np.eye(2, dtype=np.float32)
        self.last_measurement = None
        self.prediction = None

    def update(self, measurement):
        if self.last_measurement is None:
            self.last_measurement = measurement
            self.kalman_filter.statePost = np.array([measurement[0], measurement[1], 0, 0], dtype=np.float32)
        else:
            self.kalman_filter.correct(measurement)
            self.kalman_filter.predict()
        self.prediction = self.kalman_filter.statePost[:2]

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

def detect_lanes(image, lane_tracker):
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

    # Convert lines to measurement format (average of left and right lane lines)
    if lines is not None:
        left_lane, right_lane = [], []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.5:  # Ignore near-horizontal lines
                    if slope < 0:
                        left_lane.append((x1, y1))
                        left_lane.append((x2, y2))
                    else:
                        right_lane.append((x1, y1))
                        right_lane.append((x2, y2))

        if left_lane:
            left_lane_avg = np.mean(left_lane, axis=0, dtype=np.int32)
            lane_tracker.update(left_lane_avg)

        if right_lane:
            right_lane_avg = np.mean(right_lane, axis=0, dtype=np.int32)
            lane_tracker.update(right_lane_avg)

    # Draw the tracked lanes on the image
    if lane_tracker.prediction is not None:
        cv2.circle(image, tuple(lane_tracker.prediction), 5, (0, 255, 0), -1)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result

# Test the lane detection and tracking on a video stream or an image
cap = cv2.VideoCapture(r'G:\Driver Safety\Driver_Safety\test_video\test.mp4')  # Use 0 for webcam
lane_tracker = LaneTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_lanes = detect_lanes(frame, lane_tracker)
    cv2.imshow('Lane Detection with Tracking', frame_with_lanes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
