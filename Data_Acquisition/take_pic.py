# this is a helper code used to create a dataset click s to save a sample image
# run this on your rpi

import cv2
import os
import time
from datetime import datetime
from picamera2 import Picamera2

# Create output folder
output_folder = "/home/yacine/Pictures"
os.makedirs(output_folder, exist_ok=True)

# Start OpenCV thread for display
cv2.startWindowThread()

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888'}))
picam2.start()

cont = True

while True:
    # Capture current frame
    image = picam2.capture_array()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera", image)
    key = cv2.waitKey(1) & 0xFF
    # Save if 's' is pressed
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"image_{timestamp}.jpg")
        cv2.imwrite(filename, image)
        print(f"Captured: {filename}")
    # Exit if 'q' is pressed
    elif key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
