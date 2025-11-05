
import face_recognition
import os

import numpy as np

print(f"face_recognition version: {face_recognition.__version__}")
print(f"face_recognition path: {os.path.dirname(face_recognition.__file__)}")
import dlib
print(f"dlib version is {dlib.__version__}")


# pip install --no-cache-dir --force-reinstall face-recognition==1.3.0


import cv2
import tkinter as tk

# Step 1: Get screen resolution using tkinter
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()
print(f'screen_width={screen_width}')
print(f'screen_height={screen_height}')

# Step 2: Your frame or window
window_name = 'Register New User'
frame = cv2.imread(r'E:\GKM\Projects\Face Recognition Attendance System\Face Attendance system - gkmapps\employee_images\Gaja\01.jpg')  # or captured frame

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Step 3: Get frame size
frame_height, frame_width = frame.shape[:2]
print(f'frame_width={frame_width}')
print(f'frame_height={frame_height}')

# Step 4: Calculate center position
x = (screen_width - frame_width) // 2
y = (screen_height - frame_height) // 2

# Step 5: Move window to center
cv2.moveWindow(window_name, x, y)

# Step 6: Show window
#cv2.imshow(window_name, frame)
#cv2.waitKey(0)
cv2.destroyAllWindows()



# Set values manually since you already know them
screen_width = 1536
screen_height = 864
frame_width = 186
frame_height = 186

# Calculate centered position
x = (screen_width - frame_width) // 2
y = (screen_height - frame_height) // 2

# Dummy black image
frame = 255 * np.ones((frame_height, frame_width, 3), dtype=np.uint8)
frame = cv2.imread(r'E:\GKM\Projects\Face Recognition Attendance System\Face Attendance system - gkmapps\employee_images\Gaja\01.jpg')  # or captured frame

window_name = 'Register New User'

# Must call namedWindow BEFORE moveWindow
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, frame_width, frame_height)
cv2.moveWindow(window_name, x, y)

cv2.imshow(window_name, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
