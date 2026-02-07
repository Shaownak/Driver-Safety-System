import cv2
import numpy as np

width, height = 640, 480  # Adjust the dimensions as needed
image = np.zeros((height, width, 3), dtype=np.uint8)


# if in_lane or in_depth:
text = "The driver is safe"
# else:
#     text = "The driver is safe"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_thickness = 2
font_color = (255, 255, 255)  # White color
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
text_x = int((width - text_size[0]) / 2)
text_y = int((height + text_size[1]) / 2)
cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

# Display the image
cv2.imshow('Driver Safety Monitor', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

print(text)